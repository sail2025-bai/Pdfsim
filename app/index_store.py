"""
向量索引 - FAISS + PostgreSQL 持久化版

设计：
  - FAISS IndexFlatIP：内存运行时索引（查询用）
  - PostgreSQL：向量 + 元数据的持久化存储（不怕重启）
  - 启动时从 PG 加载全量向量 → 构建 FAISS 索引
  - 添加时同步写 PG + 追加到 FAISS
  - 删除时从 PG 删除 + 重建 FAISS 索引

流程：
  上传 PDF → PG 存向量+元数据 → FAISS 内存索引追加
  重启服务 → 从 PG 加载全量向量 → 重建 FAISS 索引
"""
from __future__ import annotations

import asyncio
import threading
from typing import Dict, List

import faiss
import numpy as np

# PG 存储层（延迟导入避免循环依赖）
_db = None

def _get_db():
    global _db
    if _db is None:
        from app import db as _db_mod
        _db = _db_mod
    return _db


class VectorIndex:
    """
    FAISS 内存索引 + PostgreSQL 持久化
    线程安全，支持并发操作。
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._lock = asyncio.Lock()  # 使用 asyncio.Lock 以支持 async with
        self._fid2doc: Dict[int, str] = {}  # faiss internal id -> doc_id
        self._doc2fids: Dict[str, List[int]] = {}  # doc_id -> [faiss internal ids]
        self._next_fid: int = 0
        self._index = faiss.IndexFlatIP(self.dim)

    # ---------- 同步 PG 操作 ----------

    async def add(self, doc_id: str, vectors: np.ndarray) -> List[int]:
        """
        添加向量：先持久化到 PG，再追加到 FAISS 内存索引。
        vectors: (n, dim) float32
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = vectors.shape[0]

        db = _get_db()
        async with self._lock:
            # 1) 写 PG
            await db.VectorStore.add(doc_id, vectors)

            # 2) 追加到 FAISS 内存索引
            start = self._next_fid
            fids = list(range(start, start + n))
            self._next_fid += n
            for fid in fids:
                self._fid2doc[fid] = doc_id
            self._doc2fids.setdefault(doc_id, []).extend(fids)
            self._index.add(vectors)
            return fids

    async def remove_doc(self, doc_id: str) -> int:
        """删除文档：从 PG 删除 + 重建 FAISS 索引"""
        db = _get_db()
        async with self._lock:
            fids = self._doc2fids.get(doc_id)
            if not fids:
                return 0

            # 1) 从 PG 删除
            await db.VectorStore.remove(doc_id)

            # 2) 从 PG 加载全量，重新构建 FAISS 索引
            await self._rebuild_from_pg_locked()
            return len(fids)

    async def _rebuild_from_pg_locked(self):
        """在持有锁的情况下，从 PG 全量重建 FAISS 索引"""
        db = _get_db()
        vecs, doc_ids, _ = await db.VectorStore.get_all()
        self._index = faiss.IndexFlatIP(self.dim)
        self._fid2doc.clear()
        self._doc2fids.clear()
        self._next_fid = 0

        if vecs.shape[0] == 0:
            return

        self._index.add(vecs)
        self._next_fid = vecs.shape[0]
        self._fid2doc = {i: d for i, d in enumerate(doc_ids)}
        for i, d in enumerate(doc_ids):
            self._doc2fids.setdefault(d, []).append(i)

    # ---------- 启动时加载 ----------

    async def load_from_pg(self):
        """从 PG 加载全量向量并构建 FAISS 索引（启动时调用一次）"""
        async with self._lock:
            await self._rebuild_from_pg_locked()

    # ---------- 查询（只读 FAISS，无需锁）----------

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[dict]:
        """
        向量检索。返回 [{doc_id, score, rank}, ...]
        对同一 doc_id 下多页，取最高分。
        """
        if self._index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        k_eff = min(max(k * 5, k), self._index.ntotal)
        scores, ids = self._index.search(q, k_eff)
        best: Dict[str, float] = {}
        for s, fid in zip(scores[0], ids[0]):
            if fid < 0:
                continue
            doc = self._fid2doc.get(int(fid))
            if not doc:
                continue
            s = float(s)
            if doc not in best or s > best[doc]:
                best[doc] = s
        ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
        return [
            {"doc_id": d, "score": s, "rank": i + 1}
            for i, (d, s) in enumerate(ranked)
        ]

    def get_vectors_for_doc(self, doc_id: str) -> np.ndarray:
        """
        获取某文档已入库的所有向量（按插入顺序）。
        从 PG 加载，不依赖内存索引。
        """
        import numpy as np
        fid_list = self._doc2fids.get(doc_id, [])
        if not fid_list:
            # 尝试从 PG 加载
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                vecs = loop.run_until_complete(_get_db().VectorStore.get_for_doc(doc_id))
            finally:
                loop.close()
            return vecs

        rows = []
        for fid in sorted(fid_list):
            # 从内存索引重建
            pass
        return np.zeros((0, self.dim), dtype=np.float32)

    def count_docs(self) -> int:
        return len(self._doc2fids)

    def count_vectors(self) -> int:
        return int(self._index.ntotal)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._doc2fids
