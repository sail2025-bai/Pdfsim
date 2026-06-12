"""
PDF 图纸相似性查找服务 — FastAPI 对外入口（PG 持久化版）

评分权重：
  - 图像结构：70%
  - OCR全文：20%
  - 关键字段：10%

数据存储：
  - PostgreSQL：向量 + 元数据持久化（不怕重启）
  - FAISS：内存运行时索引（查询用，启动时从 PG 加载）
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import (
    DocRecord,
    DocStore,
    VectorStore,
    close_pool,
    init_pool,
)
from app.feature_extractor import (
    FeatureExtractor,
    field_similarity,
    fusion_score,
    phash_hex,
    text_similarity,
)
from app.index_store import VectorIndex
from app.pdf_reader import render_pdf, thumbnail
from app.schemas import (
    DocInfoOut,
    HealthResponse,
    ListResponse,
    SearchResponse,
    SearchResult,
    UploadResponse,
)

_index: Optional[VectorIndex] = None


# =====================
# Lifespan（启动/关闭）
# =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    await init_pool()
    global _index
    fe = FeatureExtractor()
    _index = VectorIndex(dim=fe.dim)
    await _index.load_from_pg()
    print(f"[startup] FAISS 索引已从 PG 加载，共 {_index.count_vectors()} 个向量")
    yield
    # 关闭时
    await close_pool()
    print("[shutdown] 数据库连接已关闭")


app = FastAPI(
    title="PDF 图纸相似性查找 API (PG持久化版)",
    version="3.0.0",
    description="基于视觉特征 + OCR + 工程关键字段的融合相似度检索（向量存PostgreSQL）",
    lifespan=lifespan,
)


# =====================
# 依赖注入
# =====================

def extractor() -> FeatureExtractor:
    return FeatureExtractor()

def index() -> VectorIndex:
    if _index is None:
        raise RuntimeError("索引未初始化")
    return _index


# =====================
# 工具函数
# =====================

def _info_out(info: DocRecord) -> DocInfoOut:
    return DocInfoOut(
        doc_id=info.doc_id,
        filename=info.filename,
        size_bytes=info.size_bytes,
        num_pages=info.num_pages,
        created_at=info.created_at,
        phash=info.phash or None,
        ocr_text=info.ocr_text or None,
        text_source=info.text_source or None,
        ocr_used=info.ocr_used,
        material=info.material or None,
        process_text=info.process_text or None,
        surface_text=info.surface_text or None,
        tolerance_text=info.tolerance_text or None,
        dimension_text=info.dimension_text or None,
    )


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """直接提取 PDF 原生文本"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text("text") or ""
        texts.append(text.strip())
    doc.close()
    return "\n".join(texts)


async def _ingest_pdf(pdf_bytes: bytes, filename: str, allow_existing: bool = True):
    """
    解析 PDF -> 提取图像/OCR/字段 -> 存 PG + 建 FAISS 索引
    """
    t0 = time.time()
    try:
        doc = render_pdf(pdf_bytes, filename=filename, dpi=settings.render_dpi)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")

    if not doc.pages:
        raise HTTPException(status_code=400, detail="PDF 无可用页面")

    fe = extractor()
    idx = index()

    first_sig = fe.file_signature(doc.pages[0].image) if doc.pages else ""
    existing = await DocStore.find_by_signature(first_sig) if first_sig else None
    if existing and not allow_existing:
        raise HTTPException(status_code=409, detail=f"该PDF已存在 (doc_id={existing.doc_id})")
    if existing:
        return existing, 0, time.time() - t0

    # 提取图像特征（每页）
    vectors = []
    for page in doc.pages:
        vectors.append(fe.extract_image(page.image))
    vec_mat = np.stack(vectors, axis=0).astype(np.float32)
    doc_id = DocRecord.new_id()

    # 提取文本信息
    direct_text = _extract_pdf_text(pdf_bytes)
    ocr_text = ""
    text_source = "pdf_text"
    ocr_used = False

    if len(direct_text.strip()) < 30:
        ocr_texts = []
        for page in doc.pages[:3]:
            img_arr = np.array(page.image.convert("RGB"))
            ocr = fe.extract_ocr(img_arr)
            if ocr:
                ocr_texts.append(ocr)
        if ocr_texts:
            ocr_text = "\n".join(ocr_texts)
            text_source = "ocr"
            ocr_used = True

    full_text = direct_text if not ocr_used else ocr_text
    fields = fe.extract_fields(full_text)

    info = DocRecord(
        doc_id=doc_id,
        filename=filename,
        size_bytes=len(pdf_bytes),
        num_pages=doc.num_pages,
        created_at=time.time(),
        phash=phash_hex(doc.pages[0].image),
        signature=first_sig,
        ocr_text=full_text[:5000],
        text_source=text_source,
        ocr_used=ocr_used,
        material=fields.get("material", ""),
        process_text=fields.get("process_text", ""),
        surface_text=fields.get("surface_text", ""),
        tolerance_text=fields.get("tolerance_text", ""),
        dimension_text=fields.get("dimension_text", ""),
    )

    # 同步写 PG + FAISS
    await DocStore.add(info)
    await idx.add(doc_id, vec_mat)

    return info, len(vectors), time.time() - t0


# =====================
# API 路由
# =====================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    cnt = await DocStore.count()
    return HealthResponse(
        ok=True,
        docs=cnt,
        vectors=index().count_vectors(),
        feature_dim=extractor().dim,
    )


@app.post("/api/v1/documents", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF 文件"),
    allow_existing: bool = Form(True, description="若已存在是否直接返回已有记录"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件过大，最大 {settings.max_upload_mb} MB")
    info, n_vec, dt = await _ingest_pdf(raw, file.filename, allow_existing)
    return UploadResponse(
        ok=True, doc_id=info.doc_id, num_pages=info.num_pages,
        num_vectors=n_vec, message=f"入库耗时 {dt:.2f}s",
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def search_similar(
    file: UploadFile = File(..., description="待查询的 PDF"),
    top_k: int = Form(5, ge=1, le=50, description="返回最相似的前 N 个"),
    weight_image: float = Form(0.70, ge=0, le=1, description="图像权重"),
    weight_text: float = Form(0.20, ge=0, le=1, description="文本权重"),
    weight_field: float = Form(0.10, ge=0, le=1, description="字段权重"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")

    try:
        doc = render_pdf(raw, filename=file.filename, dpi=settings.render_dpi)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")

    if not doc.pages:
        raise HTTPException(status_code=400, detail="PDF 无可用页面")

    fe = extractor()
    idx = index()

    # 提取文本和字段
    direct_text = _extract_pdf_text(raw)
    if len(direct_text.strip()) < 30:
        ocr_texts = []
        for page in doc.pages[:3]:
            img_arr = np.array(page.image.convert("RGB"))
            ocr = fe.extract_ocr(img_arr)
            if ocr:
                ocr_texts.append(ocr)
        query_text = "\n".join(ocr_texts) if ocr_texts else direct_text
    else:
        query_text = direct_text
    query_fields = fe.extract_fields(query_text)

    # 多页检索，合并结果
    merged: Dict[str, dict] = {}
    for page in doc.pages:
        qv = fe.extract_image(page.image)
        hits = idx.search(qv, k=top_k * 3)
        for hit in hits:
            did = hit["doc_id"]
            image_sim = hit["score"]
            if did not in merged or image_sim > merged[did].get("image_sim", 0):
                merged[did] = {"image_sim": image_sim}

    # 计算融合评分
    results = []
    for did, data in merged.items():
        target_info = await DocStore.get(did)
        if not target_info:
            continue
        t_sim = text_similarity(query_text, target_info.ocr_text or "")
        f_sim = field_similarity(query_fields, await DocStore.get_meta_dict(did))
        final_score = fusion_score(data["image_sim"], t_sim, f_sim, weight_image, weight_text, weight_field)
        results.append({
            "doc_id": did,
            "score": final_score,
            "image_sim": data["image_sim"],
            "text_sim": t_sim,
            "field_sim": f_sim,
            "meta": target_info,
        })

    results.sort(key=lambda x: -x["score"])
    final = []
    for i, r in enumerate(results[:top_k]):
        final.append(SearchResult(
            doc_id=r["doc_id"],
            score=round(r["score"], 4),
            rank=i + 1,
            image_similarity=round(r["image_sim"], 4),
            text_similarity=round(r["text_sim"], 4),
            field_similarity=round(r["field_sim"], 4),
            meta=_info_out(r["meta"]) if r["meta"] else None,
        ))

    return SearchResponse(ok=True, query_doc_id=None, num_pages=doc.num_pages, results=final)


@app.post("/api/v1/search/{doc_id}", response_model=SearchResponse, tags=["Search"])
async def search_by_doc_id(doc_id: str, top_k: int = 5):
    """根据库中已有文档 ID 查找相似图纸（直接用已存向量，无需 PDF 文件）"""
    info = await DocStore.get(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="doc_id 不存在")

    idx = index()
    # 从 FAISS 索引中取出该文档的向量（内存中已有）
    fid_list = idx._doc2fids.get(doc_id, [])
    if not fid_list:
        raise HTTPException(status_code=404, detail="该文档向量不在索引中")

    # 直接从 FAISS 索引中提取该文档的向量
    import numpy as np
    vecs = idx._index.reconstruct_n(min(fid_list), len(fid_list))
    # 排序确保按 page_index 顺序
    vecs = np.array([vecs[i] for i in sorted(range(len(fid_list)), key=lambda i: fid_list[i])])

    query_text = info.ocr_text or ""
    query_fields = await DocStore.get_meta_dict(doc_id)

    merged: Dict[str, dict] = {}
    for v in vecs:
        hits = idx.search(v, k=top_k * 3)
        for hit in hits:
            did = hit["doc_id"]
            if did == doc_id:
                continue
            image_sim = hit["score"]
            if did not in merged or image_sim > merged[did].get("image_sim", 0):
                merged[did] = {"image_sim": image_sim}

    results = []
    for did, data in merged.items():
        target_info = await DocStore.get(did)
        if not target_info:
            continue
        t_sim = text_similarity(query_text, target_info.ocr_text or "")
        f_sim = field_similarity(query_fields, await DocStore.get_meta_dict(did))
        final_score = fusion_score(data["image_sim"], t_sim, f_sim)
        results.append({
            "doc_id": did,
            "score": final_score,
            "image_sim": data["image_sim"],
            "text_sim": t_sim,
            "field_sim": f_sim,
            "meta": target_info,
        })

    results.sort(key=lambda x: -x["score"])
    final = []
    for i, r in enumerate(results[:top_k]):
        final.append(SearchResult(
            doc_id=r["doc_id"],
            score=round(r["score"], 4),
            rank=i + 1,
            image_similarity=round(r["image_sim"], 4),
            text_similarity=round(r["text_sim"], 4),
            field_similarity=round(r["field_sim"], 4),
            meta=_info_out(r["meta"]) if r["meta"] else None,
        ))

    return SearchResponse(ok=True, query_doc_id=doc_id, num_pages=info.num_pages, results=final)


@app.get("/api/v1/documents", response_model=ListResponse, tags=["Documents"])
async def list_documents(skip: int = 0, limit: int = 100):
    items = await DocStore.list_all(skip=skip, limit=limit)
    total = await DocStore.count()
    return ListResponse(ok=True, total=total, items=[_info_out(x) for x in items])


@app.get("/api/v1/documents/{doc_id}", response_model=DocInfoOut, tags=["Documents"])
async def get_document(doc_id: str):
    info = await DocStore.get(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="doc_id 不存在")
    return _info_out(info)


@app.delete("/api/v1/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    info = await DocStore.get(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="doc_id 不存在")
    removed = await index().remove_doc(doc_id)
    await DocStore.remove(doc_id)
    return JSONResponse(
        status_code=200,
        content={"ok": True, "doc_id": doc_id, "vectors_removed": removed},
    )


@app.get("/", tags=["Root"])
def root():
    return {
        "name": "PDF Drawings Similarity API (PG持久化版)",
        "version": "3.0.0",
        "docs": "/docs",
        "scoring_weights": {"image_structure": 0.70, "ocr_text": 0.20, "keyword_fields": 0.10},
        "storage": "PostgreSQL (向量 + 元数据) + FAISS (内存索引)",
        "endpoints": {
            "upload": "POST /api/v1/documents",
            "search_by_file": "POST /api/v1/search",
            "search_by_id": "POST /api/v1/search/{doc_id}",
            "list": "GET /api/v1/documents",
            "delete": "DELETE /api/v1/documents/{doc_id}",
            "health": "GET /api/v1/health",
        },
    }
