"""
FastAPI 请求 / 响应 数据结构定义 - 融合版
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocInfoOut(BaseModel):
    doc_id: str
    filename: str
    size_bytes: int
    num_pages: int
    created_at: float
    phash: Optional[str] = None
    ocr_text: Optional[str] = None
    text_source: Optional[str] = None
    ocr_used: Optional[bool] = None
    material: Optional[str] = None
    process_text: Optional[str] = None
    surface_text: Optional[str] = None
    tolerance_text: Optional[str] = None
    dimension_text: Optional[str] = None

    @property
    def created_at_str(self) -> str:
        return datetime.utcfromtimestamp(self.created_at).isoformat() + "Z"


class UploadResponse(BaseModel):
    ok: bool = True
    doc_id: str
    num_pages: int
    num_vectors: int
    message: str = ""


class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    image_similarity: Optional[float] = None
    text_similarity: Optional[float] = None
    field_similarity: Optional[float] = None
    meta: Optional[DocInfoOut] = None


class SearchResponse(BaseModel):
    ok: bool = True
    query_doc_id: Optional[str] = None
    num_pages: int
    results: List[SearchResult]


class ListResponse(BaseModel):
    ok: bool = True
    total: int
    items: List[DocInfoOut]


class HealthResponse(BaseModel):
    ok: bool = True
    docs: int
    vectors: int
    feature_dim: int


class ErrorResponse(BaseModel):
    detail: str
