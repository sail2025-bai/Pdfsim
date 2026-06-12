"""
PostgreSQL + pgvector 数据库存储层

设计：
  - drawing_docs 表：元数据（文件名、OCR文本、材料、工艺等）
  - drawing_vectors 表：每页特征向量（pgvector vector(512) 类型）
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
        )


# =====================
# 数据库连接池
# =====================

_pool: Optional[asyncpg.Pool] = None


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
                doc_id        VARCHAR(24) PRIMARY KEY,
                filename      VARCHAR(512),
                size_bytes    INTEGER DEFAULT 0,
                num_pages     INTEGER DEFAULT 0,
                created_at    DOUBLE PRECISION,
                phash         VARCHAR(128),
                signature     VARCHAR(128),
                ocr_text      TEXT,
                text_source   VARCHAR(32) DEFAULT 'pdf_text',
                ocr_used      BOOLEAN DEFAULT FALSE,
                material      VARCHAR(512),
                process_text  VARCHAR(512),
                surface_text  VARCHAR(512),
                tolerance_text VARCHAR(512),
                dimension_text VARCHAR(1024),
                extra         JSONB DEFAULT '{}'
            )
        """)

        # 向量表（pgvector vector 类型）
        dim = settings.feature_dim
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS drawing_vectors (
                doc_id     VARCHAR(24) REFERENCES drawing_docs(doc_id) ON DELETE CASCADE,
                page_index INTEGER,
                embedding  vector({dim}) NOT NULL,
                PRIMARY KEY (doc_id, page_index)
            )
        """)

        # 索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id ON drawing_vectors(doc_id)
        """)

        # pgvector HNSW 索引（向量量大时加速查询）
        # 先尝试创建，如果已存在则忽略
        try:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_vectors_embedding
                ON drawing_vectors
                USING hnsw (embedding vector_cosine_ops)
            """)
        except Exception:
            pass  # 索引创建可能因数据量不足而失败，忽略

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
                 material, process_text, surface_text, tolerance_text, dimension_text, extra)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
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
                    extra=EXCLUDED.extra
            """,
                info.doc_id, info.filename, info.size_bytes, info.num_pages,
                info.created_at, info.phash, info.signature, info.ocr_text,
                info.text_source, info.ocr_used, info.material, info.process_text,
                info.surface_text, info.tolerance_text, info.dimension_text,
                json.dumps(info.extra),
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
            # r = "DELETE N"
            return int(r.split()[-1]) if r.startswith("DELETE") else 0

    @staticmethod
    async def search(
        query_vec: np.ndarray,
        k: int = 10,
        exclude_doc_id: Optional[str] = None,
    ) -> List[dict]:
        """
        向量相似度搜索（余弦距离）
        返回 [{doc_id, page_index, similarity}, ...]
        对同一 doc_id 取最高相似度
        """
        p = pool()
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        async with p.acquire() as conn:
            if exclude_doc_id:
                rows = await conn.fetch(
                    """
                    SELECT doc_id, page_index, 1 - (embedding <=> $1) AS similarity
                    FROM drawing_vectors
                    WHERE doc_id != $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                    """,
                    query_vec, exclude_doc_id, k * 5,
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

        # 按相似度排序
        ranked = sorted(best.values(), key=lambda x: -x["similarity"])[:k]
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        return ranked

    @staticmethod
    async def multi_page_search(
        query_vectors: np.ndarray,
        k: int = 10,
        exclude_doc_id: Optional[str] = None,
    ) -> List[dict]:
        """
        多页查询：对每页向量搜索，合并结果取每个 doc_id 的最高相似度
        """
        merged: Dict[str, dict] = {}
        for qv in query_vectors:
            hits = await VectorStore.search(qv, k=k * 2, exclude_doc_id=exclude_doc_id)
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
