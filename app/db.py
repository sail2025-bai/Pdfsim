"""
PostgreSQL + pgvector 数据库存储层

设计：
  - drawing_docs 表：元数据（文件名、OCR文本、材料、工艺等）
  - drawing_vectors 表：每页特征向量（pgvector vector(512) 类型）
  - drawing_import_status 表：图纸入库状态追踪（不侵入氚云原数据）
  - 向量搜索直接用 pgvector 的 <=> 余弦距离操作符
  - 无需 FAISS，重启零成本
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.config import settings


# =====================
# 数据模型
# =====================

@dataclass
class DocRecord:
    doc_id: str
    filename: str = ""
    size_bytes: int = 0
    num_pages: int = 0
    created_at: float = 0.0
    phash: str = ""
    signature: str = ""
    ocr_text: str = ""
    text_source: str = "pdf_text"
    ocr_used: bool = False
    material: str = ""
    process_text: str = ""
    surface_text: str = ""
    tolerance_text: str = ""
    dimension_text: str = ""
    extra: dict = field(default_factory=dict)
    # 氚云关联
    h3yun_object_id: str = ""
    h3yun_schema_code: str = ""

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:24]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "num_pages": self.num_pages,
            "created_at": self.created_at,
            "phash": self.phash,
            "signature": self.signature,
            "ocr_text": self.ocr_text,
            "text_source": self.text_source,
            "ocr_used": self.ocr_used,
            "material": self.material,
            "process_text": self.process_text,
            "surface_text": self.surface_text,
            "tolerance_text": self.tolerance_text,
            "dimension_text": self.dimension_text,
            "extra": self.extra,
            "h3yun_object_id": self.h3yun_object_id,
            "h3yun_schema_code": self.h3yun_schema_code,
        }

    @classmethod
    def from_row(cls, row) -> "DocRecord":
        return cls(
            doc_id=row["doc_id"],
            filename=row.get("filename", ""),
            size_bytes=int(row.get("size_bytes", 0)),
            num_pages=int(row.get("num_pages", 0)),
            created_at=float(row.get("created_at", 0)),
            phash=row.get("phash", ""),
            signature=row.get("signature", ""),
            ocr_text=row.get("ocr_text", ""),
            text_source=row.get("text_source", "pdf_text"),
            ocr_used=bool(row.get("ocr_used", False)),
            material=row.get("material", ""),
            process_text=row.get("process_text", ""),
            surface_text=row.get("surface_text", ""),
            tolerance_text=row.get("tolerance_text", ""),
            dimension_text=row.get("dimension_text", ""),
            extra=json.loads(row["extra"]) if row.get("extra") else {},
            h3yun_object_id=row.get("h3yun_object_id", ""),
            h3yun_schema_code=row.get("h3yun_schema_code", ""),
        )


# 入库状态枚举
IMPORT_PENDING = "pending"       # 待入库
IMPORT_INGESTING = "ingesting"   # 入库中
IMPORT_DONE = "done"             # 已入库
IMPORT_FAILED = "failed"         # 入库失败
IMPORT_SKIP = "skipped"          # 跳过（非PDF等）
IMPORT_DEAD = "dead"             # 重试耗尽，需人工介入

MAX_RETRY = 3  # 最大重试次数


@dataclass
class ImportStatus:
    """图纸入库状态记录"""
    id: int = 0
    h3yun_object_id: str = ""
    h3yun_schema_code: str = ""
    doc_id: str = ""
    filename: str = ""
    status: str = IMPORT_PENDING
    error_message: str = ""
    attachment_info: str = ""
    retry_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row) -> "ImportStatus":
        return cls(
            id=int(row.get("id", 0)),
            h3yun_object_id=row.get("h3yun_object_id", ""),
            h3yun_schema_code=row.get("h3yun_schema_code", ""),
            doc_id=row.get("doc_id", ""),
            filename=row.get("filename", ""),
            status=row.get("status", IMPORT_PENDING),
            error_message=row.get("error_message", ""),
            attachment_info=row.get("attachment_info", ""),
            retry_count=int(row.get("retry_count", 0)),
            created_at=float(row.get("created_at", 0)),
            updated_at=float(row.get("updated_at", 0)),
        )


# =====================
# 数据库连接池
# =====================

_pool: Optional[asyncpg.Pool] = None


async def _check_vector_dim(conn) -> Optional[int]:
    """检查 drawing_vectors 表的向量维度，不存在返回 None"""
    try:
        # 检查表是否存在
        exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'drawing_vectors' AND table_schema = 'public'
            )
        """)
        if not exists:
            return None
        # 从已有数据推断维度（最可靠）
        sample = await conn.fetchrow(
            "SELECT embedding FROM drawing_vectors LIMIT 1"
        )
        if sample:
            return len(sample["embedding"])
        # 表存在但无数据，从DDL解析
        # pgvector 存维度在 typmod 中: dimension = (typmod - 4) / 4
        # 参考: https://github.com/pgvector/pgvector/blob/master/src/vector.c
        typmod = await conn.fetchval("""
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'drawing_vectors' AND a.attname = 'embedding'
        """)
        if typmod and typmod > 0:
            # pgvector typmod 编码: (dims << 16) | ndims，实际维度 = typmod >> 16
            # 但更简单的计算: (typmod - VARHDRSZ) 直接就是维度
            # VARHDRSZ = 4 for pgvector
            return (typmod - 4) // 4
        return None
    except Exception:
        return None


async def init_pool():
    """初始化连接池、注册 pgvector 扩展、创建表"""
    global _pool
    if _pool is not None:
        return _pool

    cfg = settings.db

    async def _init_conn(conn):
        """每个连接注册 pgvector 类型"""
        await register_vector(conn)

    _pool = await asyncpg.create_pool(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
        min_size=2,
        max_size=cfg.max_connections,
        init=_init_conn,
    )

    async with _pool.acquire() as conn:
        # 启用 pgvector 扩展
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # 启用 uuid-ossp
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

        # 文档元数据表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS drawing_docs (
                doc_id            VARCHAR(24) PRIMARY KEY,
                filename          VARCHAR(512),
                size_bytes        INTEGER DEFAULT 0,
                num_pages         INTEGER DEFAULT 0,
                created_at        DOUBLE PRECISION,
                phash             VARCHAR(128),
                signature         VARCHAR(128),
                ocr_text          TEXT,
                text_source       VARCHAR(32) DEFAULT 'pdf_text',
                ocr_used          BOOLEAN DEFAULT FALSE,
                material          VARCHAR(512),
                process_text      VARCHAR(512),
                surface_text      VARCHAR(512),
                tolerance_text    VARCHAR(512),
                dimension_text    VARCHAR(1024),
                extra             JSONB DEFAULT '{}',
                h3yun_object_id   VARCHAR(64) DEFAULT '',
                h3yun_schema_code VARCHAR(64) DEFAULT ''
            )
        """)

        # 向量表（pgvector vector 类型）
        dim = settings.feature_dim
        # 检查已有表的向量维度，不一致则重建（512→768等升级场景）
        existing_dim = await _check_vector_dim(conn)
        if existing_dim is not None and existing_dim != dim:
            print(f"[migration] drawing_vectors 维度 {existing_dim} → {dim}，重建表")
            await conn.execute("DROP INDEX IF EXISTS idx_vectors_embedding")
            await conn.execute("DROP TABLE drawing_vectors")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS drawing_vectors (
                doc_id     VARCHAR(24) REFERENCES drawing_docs(doc_id) ON DELETE CASCADE,
                page_index INTEGER,
                embedding  vector({dim}) NOT NULL,
                PRIMARY KEY (doc_id, page_index)
            )
        """)

        # 入库状态追踪表（不侵入氚云原数据）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS drawing_import_status (
                id                SERIAL PRIMARY KEY,
                h3yun_object_id   VARCHAR(64) NOT NULL,
                h3yun_schema_code VARCHAR(64) DEFAULT '',
                doc_id            VARCHAR(24) DEFAULT '',
                filename          VARCHAR(512) DEFAULT '',
                status            VARCHAR(32) DEFAULT 'pending',
                error_message     TEXT DEFAULT '',
                attachment_info   VARCHAR(256) DEFAULT '',
                retry_count       INTEGER DEFAULT 0,
                created_at        DOUBLE PRECISION DEFAULT 0,
                updated_at        DOUBLE PRECISION DEFAULT 0
            )
        """)

        # 索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id ON drawing_vectors(doc_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_import_h3yun ON drawing_import_status(h3yun_object_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_import_status ON drawing_import_status(status)
        """)

        # pgvector HNSW 索引
        try:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_vectors_embedding
                ON drawing_vectors
                USING hnsw (embedding vector_cosine_ops)
            """)
        except Exception:
            pass

    return _pool


async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("数据库未初始化，请先调用 init_pool()")
    return _pool


# =====================
# DocStore: 元数据 CRUD
# =====================

class DocStore:
    """PostgreSQL 元数据存储"""

    @staticmethod
    async def add(info: DocRecord) -> DocRecord:
        p = pool()
        async with p.acquire() as conn:
            await conn.execute("""
                INSERT INTO drawing_docs
                (doc_id, filename, size_bytes, num_pages, created_at,
                 phash, signature, ocr_text, text_source, ocr_used,
                 material, process_text, surface_text, tolerance_text, dimension_text,
                 extra, h3yun_object_id, h3yun_schema_code)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
                ON CONFLICT (doc_id) DO UPDATE SET
                    filename=EXCLUDED.filename,
                    size_bytes=EXCLUDED.size_bytes,
                    num_pages=EXCLUDED.num_pages,
                    ocr_text=EXCLUDED.ocr_text,
                    material=EXCLUDED.material,
                    process_text=EXCLUDED.process_text,
                    surface_text=EXCLUDED.surface_text,
                    tolerance_text=EXCLUDED.tolerance_text,
                    dimension_text=EXCLUDED.dimension_text,
                    extra=EXCLUDED.extra,
                    h3yun_object_id=EXCLUDED.h3yun_object_id,
                    h3yun_schema_code=EXCLUDED.h3yun_schema_code
            """,
                info.doc_id, info.filename, info.size_bytes, info.num_pages,
                info.created_at, info.phash, info.signature, info.ocr_text,
                info.text_source, info.ocr_used, info.material, info.process_text,
                info.surface_text, info.tolerance_text, info.dimension_text,
                json.dumps(info.extra),
                info.h3yun_object_id, info.h3yun_schema_code,
            )
        return info

    @staticmethod
    async def get(doc_id: str) -> Optional[DocRecord]:
        p = pool()
        async with p.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM drawing_docs WHERE doc_id=$1", doc_id
            )
            return DocRecord.from_row(row) if row else None

    @staticmethod
    async def remove(doc_id: str) -> bool:
        """删除文档（向量表通过 ON DELETE CASCADE 自动删除）"""
        p = pool()
        async with p.acquire() as conn:
            r = await conn.execute(
                "DELETE FROM drawing_docs WHERE doc_id=$1", doc_id
            )
            return r != "DELETE 0"

    @staticmethod
    async def list_all(skip: int = 0, limit: int = 100) -> List[DocRecord]:
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM drawing_docs ORDER BY created_at DESC OFFSET $1 LIMIT $2",
                skip, limit,
            )
            return [DocRecord.from_row(r) for r in rows]

    @staticmethod
    async def count() -> int:
        p = pool()
        async with p.acquire() as conn:
            r = await conn.fetchrow("SELECT COUNT(*) as c FROM drawing_docs")
            return int(r["c"])

    @staticmethod
    async def find_by_signature(sig: str) -> Optional[DocRecord]:
        p = pool()
        async with p.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM drawing_docs WHERE signature=$1 LIMIT 1", sig
            )
            return DocRecord.from_row(row) if row else None

    @staticmethod
    async def find_by_h3yun_id(h3yun_object_id: str) -> Optional[DocRecord]:
        """按氚云ObjectId查找已入库文档"""
        p = pool()
        async with p.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM drawing_docs WHERE h3yun_object_id=$1 LIMIT 1",
                h3yun_object_id,
            )
            return DocRecord.from_row(row) if row else None

    @staticmethod
    async def find_doc_ids_by_h3yun_ids(h3yun_object_ids: List[str]) -> List[str]:
        """批量按氚云ObjectId查找对应的doc_id列表"""
        if not h3yun_object_ids:
            return []
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id FROM drawing_docs WHERE h3yun_object_id = ANY($1)",
                h3yun_object_ids,
            )
            return [r["doc_id"] for r in rows]

    @staticmethod
    async def get_meta_dict(doc_id: str) -> Dict[str, str]:
        info = await DocStore.get(doc_id)
        if not info:
            return {}
        return {
            "material": info.material,
            "process_text": info.process_text,
            "surface_text": info.surface_text,
            "tolerance_text": info.tolerance_text,
            "dimension_text": info.dimension_text,
        }


# =====================
# VectorStore: pgvector 向量存储与检索
# =====================

class VectorStore:
    """pgvector 向量存储"""

    @staticmethod
    async def add(doc_id: str, vectors: np.ndarray) -> List[int]:
        """插入一个文档的多页向量，返回 page_index 列表"""
        p = pool()
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        page_indices = []
        async with p.acquire() as conn:
            for i, vec in enumerate(vectors):
                await conn.execute(
                    """
                    INSERT INTO drawing_vectors (doc_id, page_index, embedding)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (doc_id, page_index) DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    doc_id, i, vec,
                )
                page_indices.append(i)
        return page_indices

    @staticmethod
    async def remove(doc_id: str) -> int:
        """删除文档的所有向量"""
        p = pool()
        async with p.acquire() as conn:
            r = await conn.execute(
                "DELETE FROM drawing_vectors WHERE doc_id=$1", doc_id
            )
            return int(r.split()[-1]) if r.startswith("DELETE") else 0

    @staticmethod
    async def search(
        query_vec: np.ndarray,
        k: int = 10,
        exclude_doc_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        向量相似度搜索（余弦距离）
        返回 [{doc_id, page_index, similarity}, ...]
        对同一 doc_id 取最高相似度
        exclude_doc_ids: 要排除的 doc_id 列表（按 h3yun_object_id 过滤自身）
        """
        p = pool()
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        async with p.acquire() as conn:
            if exclude_doc_ids:
                rows = await conn.fetch(
                    """
                    SELECT doc_id, page_index, 1 - (embedding <=> $1) AS similarity
                    FROM drawing_vectors
                    WHERE doc_id != ALL($2::varchar[])
                    ORDER BY embedding <=> $1
                    LIMIT $3
                    """,
                    query_vec, exclude_doc_ids, k * 5,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT doc_id, page_index, 1 - (embedding <=> $1) AS similarity
                    FROM drawing_vectors
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """,
                    query_vec, k * 5,
                )

        # 同一 doc_id 取最高相似度
        best: Dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            sim = float(row["similarity"])
            if did not in best or sim > best[did]["similarity"]:
                best[did] = {
                    "doc_id": did,
                    "page_index": row["page_index"],
                    "similarity": sim,
                }

        ranked = sorted(best.values(), key=lambda x: -x["similarity"])[:k]
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        return ranked

    @staticmethod
    async def multi_page_search(
        query_vectors: np.ndarray,
        k: int = 10,
        exclude_doc_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        多页查询：对每页向量搜索，合并结果取每个 doc_id 的最高相似度
        exclude_doc_ids: 要排除的 doc_id 列表
        """
        merged: Dict[str, dict] = {}
        for qv in query_vectors:
            hits = await VectorStore.search(qv, k=k * 2, exclude_doc_ids=exclude_doc_ids)
            for hit in hits:
                did = hit["doc_id"]
                if did not in merged or hit["similarity"] > merged[did]["similarity"]:
                    merged[did] = hit

        ranked = sorted(merged.values(), key=lambda x: -x["similarity"])[:k]
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        return ranked

    @staticmethod
    async def count() -> int:
        p = pool()
        async with p.acquire() as conn:
            r = await conn.fetchrow("SELECT COUNT(*) as c FROM drawing_vectors")
            return int(r["c"])

    @staticmethod
    async def count_docs() -> int:
        p = pool()
        async with p.acquire() as conn:
            r = await conn.fetchrow("SELECT COUNT(DISTINCT doc_id) as c FROM drawing_vectors")
            return int(r["c"])

    @staticmethod
    async def get_for_doc(doc_id: str) -> np.ndarray:
        """获取某文档的所有向量"""
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT embedding FROM drawing_vectors WHERE doc_id=$1 ORDER BY page_index",
                doc_id,
            )
        if not rows:
            return np.zeros((0, settings.feature_dim), dtype=np.float32)
        return np.stack([np.array(r["embedding"], dtype=np.float32) for r in rows], axis=0)


# =====================
# ImportStatusStore: 入库状态追踪（不侵入氚云原数据）
# =====================

class ImportStatusStore:
    """入库状态追踪（存我们自己的PG，不碰氚云原数据）"""

    @staticmethod
    async def upsert(
        h3yun_object_id: str,
        h3yun_schema_code: str = "",
        doc_id: str = "",
        filename: str = "",
        status: str = IMPORT_PENDING,
        error_message: str = "",
        attachment_info: str = "",
    ) -> ImportStatus:
        """插入或更新入库状态（自动管理重试计数）"""
        p = pool()
        now = time.time()
        async with p.acquire() as conn:
            # 先看当前记录
            existing = await conn.fetchrow(
                "SELECT retry_count FROM drawing_import_status WHERE h3yun_object_id=$1",
                h3yun_object_id,
            )
            current_retry = int(existing["retry_count"]) if existing else 0

            # 如果标记为 failed，累加 retry_count
            new_retry = current_retry
            actual_status = status
            if status == IMPORT_FAILED:
                new_retry = current_retry + 1
                if new_retry >= MAX_RETRY:
                    actual_status = IMPORT_DEAD  # 重试耗尽

            row = await conn.fetchrow("""
                INSERT INTO drawing_import_status
                    (h3yun_object_id, h3yun_schema_code, doc_id, filename,
                     status, error_message, attachment_info, retry_count, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                ON CONFLICT (h3yun_object_id) DO UPDATE SET
                    h3yun_schema_code = EXCLUDED.h3yun_schema_code,
                    doc_id = EXCLUDED.doc_id,
                    filename = EXCLUDED.filename,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    attachment_info = EXCLUDED.attachment_info,
                    retry_count = EXCLUDED.retry_count,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
            """,
                h3yun_object_id, h3yun_schema_code, doc_id, filename,
                actual_status, error_message, attachment_info, new_retry, now,
            )
            if not row:
                row = await conn.fetchrow(
                    "SELECT * FROM drawing_import_status WHERE h3yun_object_id=$1",
                    h3yun_object_id,
                )
            return ImportStatus.from_row(row) if row else ImportStatus()

    @staticmethod
    async def get(h3yun_object_id: str) -> Optional[ImportStatus]:
        p = pool()
        async with p.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM drawing_import_status WHERE h3yun_object_id=$1",
                h3yun_object_id,
            )
            return ImportStatus.from_row(row) if row else None

    @staticmethod
    async def list_by_status(status: str, limit: int = 100) -> List[ImportStatus]:
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM drawing_import_status WHERE status=$1 ORDER BY created_at LIMIT $2",
                status, limit,
            )
            return [ImportStatus.from_row(r) for r in rows]

    @staticmethod
    async def count_by_status() -> Dict[str, int]:
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, COUNT(*) as c FROM drawing_import_status GROUP BY status"
            )
            return {r["status"]: int(r["c"]) for r in rows}

    @staticmethod
    async def count() -> int:
        p = pool()
        async with p.acquire() as conn:
            r = await conn.fetchrow("SELECT COUNT(*) as c FROM drawing_import_status")
            return int(r["c"])

    @staticmethod
    async def get_pending(limit: int = 100) -> List[ImportStatus]:
        """获取待入库列表"""
        return await ImportStatusStore.list_by_status(IMPORT_PENDING, limit)

    @staticmethod
    async def get_failed(limit: int = 100) -> List[ImportStatus]:
        """获取失败列表（可重试）"""
        return await ImportStatusStore.list_by_status(IMPORT_FAILED, limit)
