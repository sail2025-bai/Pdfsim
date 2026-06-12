"""
索引存储（pgvector版）

本模块仅保留 VectorIndex 类作为 db.VectorStore 的轻量包装，
兼容原有 API 调用方式。实际向量搜索全部由 pgvector SQL 完成。
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import numpy as np

from app.db import VectorStore


class VectorIndex:
    """向量索引（pgvector 代理）"""

    def __init__(self, dim: int = 512, **kwargs):
        self.dim = dim

    async def add(self, doc_id: str, vectors: np.ndarray) -> List[int]:
        """添加文档向量"""
        return await VectorStore.add(doc_id, vectors)

    async def remove_doc(self, doc_id: str) -> int:
        """删除文档向量"""
        return await VectorStore.remove(doc_id)

    async def load_from_pg(self):
        """从 PG 加载（pgvector版不需要预加载，保持接口兼容）"""
        pass

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[dict]:
        """同步搜索（兼容旧 API）"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            VectorStore.search(query_vec, k=k)
        )

    async def async_search(self, query_vec: np.ndarray, k: int = 5,
                           exclude_doc_id: Optional[str] = None) -> List[dict]:
        """异步搜索"""
        return await VectorStore.search(query_vec, k=k, exclude_doc_id=exclude_doc_id)

    async def async_multi_search(self, query_vectors: np.ndarray, k: int = 5,
                                  exclude_doc_id: Optional[str] = None) -> List[dict]:
        """多页异步搜索"""
        return await VectorStore.multi_page_search(
            query_vectors, k=k, exclude_doc_id=exclude_doc_id
        )

    def count_vectors(self) -> int:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(VectorStore.count())

    def count_docs(self) -> int:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(VectorStore.count_docs())

    def __contains__(self, doc_id: str) -> bool:
        return False  # 简化，实际查询时不依赖此判断
