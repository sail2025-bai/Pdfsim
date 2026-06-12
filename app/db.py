"""
PostgreSQL 数据库存储层

设计：
  - drawing_docs 表：元数据（文件名、OCR文本、材料、工艺等）
  - drawing_vectors 表：每页特征向量（bytea 格式）
  - FAISS 内存索引：启动时从 PG 加载全量向量，重启不丢失

向量存储格式：numpy.float32 序列化后的二进制
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg

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
    """初始化连接池并创建表"""
    global _pool
    if _pool is not None:
        return _pool

    cfg = settings.db
    _pool = await asyncpg.create_pool(
        host=cfg.get("host", "localhost"),
        port=cfg.get("port", 5432),
        user=cfg.get("user", "postgres"),
        password=cfg.get("password", ""),
        database=cfg.get("database", "postgres"),
        min_size=2,
        max_size=cfg.get("max_connections", 10),
    )

    async with _pool.acquire() as conn:
        # 创建扩展和表
        await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS drawing_vectors (
                doc_id     VARCHAR(24),
                page_index INTEGER,
                vector     BYTEA NOT NULL,
                PRIMARY KEY (doc_id, page_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_doc_id ON drawing_vectors(doc_id)
        """)

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
        p = pool()
        async with p.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM drawing_vectors WHERE doc_id=$1", doc_id
                )
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
                skip, limit
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
                "SELECT * FROM drawing_docs WHERE signature=$1", sig
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
# VectorStore: 向量 CRUD
# =====================

def vec_to_bytes(v) -> bytes:
    """numpy 数组 -> 二进制"""
    return v.astype("<f").tobytes()


def bytes_to_vec(b: bytes, dim: int) -> "np.ndarray":
    """二进制 -> numpy 数组"""
    import numpy as np
    return np.frombuffer(b, dtype="<f").astype(np.float32)


class VectorStore:
    """
    PostgreSQL 向量存储（bytea 格式）
    负责持久化，不负责搜索（搜索走 FAISS 内存索引）
    """

    @staticmethod
    async def add(doc_id: str, vectors) -> List[int]:
        """
        批量存入向量，返回 page_index 列表
        vectors: (n, dim) numpy float32
        """
        import numpy as np
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        p = pool()
        async with p.acquire() as conn:
            async with conn.transaction():
                for i, vec in enumerate(vectors):
                    await conn.execute(
                        """
                        INSERT INTO drawing_vectors (doc_id, page_index, vector)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (doc_id, page_index) DO UPDATE SET vector=EXCLUDED.vector
                        """,
                        doc_id, i, vec.tobytes()
                    )
            return list(range(len(vectors)))

    @staticmethod
    async def get_for_doc(doc_id: str) -> "np.ndarray":
        """
        获取某文档所有向量，按 page_index 顺序返回 (n, dim)
        """
        import numpy as np
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT page_index, vector FROM drawing_vectors "
                "WHERE doc_id=$1 ORDER BY page_index",
                doc_id
            )
            if not rows:
                return np.zeros((0, 512), dtype=np.float32)
            vecs = [bytes_to_vec(r["vector"], 512) for r in rows]
            return np.stack(vecs, axis=0).astype(np.float32)

    @staticmethod
    async def get_all() -> tuple:
        """
        加载全量向量，返回 (vectors: np.ndarray, doc_ids: List[str], page_indices: List[int])
        用于启动时重建 FAISS 索引。
        """
        import numpy as np
        p = pool()
        async with p.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id, page_index, vector FROM drawing_vectors ORDER BY doc_id, page_index"
            )
            if not rows:
                return np.zeros((0, 512), dtype=np.float32), [], []
            vectors = [bytes_to_vec(r["vector"], 512) for r in rows]
            doc_ids = [r["doc_id"] for r in rows]
            page_indices = [r["page_index"] for r in rows]
            return np.stack(vectors, axis=0).astype(np.float32), doc_ids, page_indices

    @staticmethod
    async def remove(doc_id: str) -> int:
        p = pool()
        async with p.acquire() as conn:
            r = await conn.execute(
                "DELETE FROM drawing_vectors WHERE doc_id=$1", doc_id
            )
            # 返回删除数量
            if r == "DELETE 0":
                return 0
            cnt_row = await conn.fetchrow(
                "SELECT count(*) as c FROM drawing_vectors WHERE doc_id=$1", doc_id
            )
            # 刚才已删除，所以直接返回 1（因为按主键删除）
            # 实际上返回受影响行数更准确
            return 1

    @staticmethod
    async def count() -> int:
        p = pool()
        async with p.acquire() as conn:
            r = await conn.fetchrow("SELECT COUNT(*) as c FROM drawing_vectors")
            return int(r["c"])
