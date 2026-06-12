"""
PDF 图纸相似性查找服务 — FastAPI 对外入口（BGE多模态版）

与 v4.0 的区别：
  - 特征提取从 pHash+HOG 升级为 Visualized-BGE 768维
  - 图文联合编码：encode(image=图纸, text=OCR文本) → 768维统一向量
  - 搜索简化：一次 pgvector 向量搜索搞定，不再需要三段融合评分
  - 返回结果同时保留元数据（材料/工艺/公差等）供参考
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import (
    DocRecord,
    DocStore,
    ImportStatusStore,
    IMPORT_PENDING,
    IMPORT_INGESTING,
    IMPORT_DONE,
    IMPORT_FAILED,
    IMPORT_SKIP,
    IMPORT_DEAD,
    MAX_RETRY,
    VectorStore,
    close_pool,
    init_pool,
)
from app.feature_extractor import FeatureExtractor, cosine_similarity
from app.pdf_reader import render_pdf
from app.schemas import (
    DocInfoOut,
    HealthResponse,
    ListResponse,
    SearchResponse,
    SearchResult,
    UploadResponse,
)


# =====================
# Lifespan（启动/关闭）
# =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：初始化 PG 连接池 + pgvector 扩展 + 预加载BGE模型
    await init_pool()
    vec_count = await VectorStore.count()
    doc_count = await DocStore.count()
    # 预加载BGE模型（避免第一次请求慢）
    fe = FeatureExtractor()
    print(f"[startup] pgvector 就绪，共 {doc_count} 文档 / {vec_count} 向量，特征维度 {fe.dim}")
    yield
    await close_pool()
    print("[shutdown] 数据库连接已关闭")


app = FastAPI(
    title="PDF 图纸相似性查找 API (BGE多模态版)",
    version="5.0.0",
    description="基于 Visualized-BGE 768维图文联合嵌入 + pgvector 向量搜索",
    lifespan=lifespan,
)


# =====================
# 依赖注入
# =====================

_fe_instance: Optional[FeatureExtractor] = None

def extractor() -> FeatureExtractor:
    global _fe_instance
    if _fe_instance is None:
        _fe_instance = FeatureExtractor()
    return _fe_instance


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


async def _ingest_pdf(pdf_bytes: bytes, filename: str, allow_existing: bool = True,
                      h3yun_object_id: str = "", h3yun_schema_code: str = ""):
    """
    解析 PDF → 提取图像/OCR/字段 → BGE图文联合编码 → 存 PG + pgvector
    """
    t0 = time.time()
    try:
        doc = render_pdf(pdf_bytes, filename=filename, dpi=settings.render_dpi)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")

    if not doc.pages:
        raise HTTPException(status_code=400, detail="PDF 无可用页面")

    fe = extractor()

    # 去重检测
    first_sig = fe.file_signature(doc.pages[0].image) if doc.pages else ""
    existing = await DocStore.find_by_signature(first_sig) if first_sig else None
    if existing and not allow_existing:
        raise HTTPException(status_code=409, detail=f"该PDF已存在 (doc_id={existing.doc_id})")
    if existing:
        return existing, 0, time.time() - t0

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

    # BGE 图文联合编码（每页）
    # 核心：encode(image=图纸页, text=OCR文本) → 768维统一向量
    vectors = []
    for page in doc.pages:
        vec = fe.extract(page.image, text=full_text[:2000])  # 截断过长文本
        vectors.append(vec)
    vec_mat = np.stack(vectors, axis=0).astype(np.float32)

    doc_id = DocRecord.new_id()

    # 提取关键字段（元数据，不参与向量编码）
    fields = fe.extract_fields(full_text)

    info = DocRecord(
        doc_id=doc_id,
        filename=filename,
        size_bytes=len(pdf_bytes),
        num_pages=doc.num_pages,
        created_at=time.time(),
        phash=fe.file_signature(doc.pages[0])[:16] if doc.pages else "",
        signature=first_sig,
        ocr_text=full_text[:5000],
        text_source=text_source,
        ocr_used=ocr_used,
        material=fields.get("material", ""),
        process_text=fields.get("process_text", ""),
        surface_text=fields.get("surface_text", ""),
        tolerance_text=fields.get("tolerance_text", ""),
        dimension_text=fields.get("dimension_text", ""),
        h3yun_object_id=h3yun_object_id,
        h3yun_schema_code=h3yun_schema_code,
    )

    # 写 PG 元数据 + 向量
    await DocStore.add(info)
    await VectorStore.add(doc_id, vec_mat)

    return info, len(vectors), time.time() - t0


# =====================
# API 路由
# =====================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    doc_cnt = await DocStore.count()
    vec_cnt = await VectorStore.count()
    return HealthResponse(
        ok=True,
        docs=doc_cnt,
        vectors=vec_cnt,
        feature_dim=extractor().dim,
    )


@app.post("/api/v1/documents", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF 文件"),
    allow_existing: bool = Form(True, description="若已存在是否直接返回已有记录"),
    h3yun_object_id: str = Form("", description="氚云ObjectId（批量入库时传入）"),
    h3yun_schema_code: str = Form("", description="氚云SchemaCode"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件过大，最大 {settings.max_upload_mb} MB")
    info, n_vec, dt = await _ingest_pdf(
        raw, file.filename, allow_existing,
        h3yun_object_id=h3yun_object_id,
        h3yun_schema_code=h3yun_schema_code,
    )
    return UploadResponse(
        ok=True, doc_id=info.doc_id, num_pages=info.num_pages,
        num_vectors=n_vec, message=f"入库耗时 {dt:.2f}s",
    )


@app.get("/api/v1/import/status", tags=["Import"])
async def import_status_summary():
    """入库状态汇总"""
    counts = await ImportStatusStore.count_by_status()
    total = await ImportStatusStore.count()
    return {"ok": True, "total": total, "by_status": counts}


@app.get("/api/v1/import/pending", tags=["Import"])
async def import_pending_list(limit: int = 100):
    """获取待入库列表"""
    items = await ImportStatusStore.get_pending(limit)
    return {"ok": True, "count": len(items), "items": [
        {"h3yun_object_id": i.h3yun_object_id, "filename": i.filename, "status": i.status}
        for i in items
    ]}


@app.get("/api/v1/import/failed", tags=["Import"])
async def import_failed_list(limit: int = 100):
    """获取失败列表（可重试）"""
    items = await ImportStatusStore.get_failed(limit)
    return {"ok": True, "count": len(items), "items": [
        {"h3yun_object_id": i.h3yun_object_id, "filename": i.filename,
         "status": i.status, "error": i.error_message, "retry_count": i.retry_count}
        for i in items
    ]}


@app.post("/api/v1/import/record", tags=["Import"])
async def import_record(h3yun_object_id: str, h3yun_schema_code: str = "",
                        doc_id: str = "", filename: str = "",
                        status: str = IMPORT_PENDING, error_message: str = "",
                        attachment_info: str = ""):
    """记录或更新入库状态（供批量脚本调用，不碰氚云原数据）"""
    result = await ImportStatusStore.upsert(
        h3yun_object_id=h3yun_object_id,
        h3yun_schema_code=h3yun_schema_code,
        doc_id=doc_id,
        filename=filename,
        status=status,
        error_message=error_message,
        attachment_info=attachment_info,
    )
    return {"ok": True, "h3yun_object_id": result.h3yun_object_id,
            "status": result.status, "retry_count": result.retry_count}


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def search_similar(
    file: UploadFile = File(..., description="待查询的 PDF"),
    top_k: int = Form(10, ge=1, le=50, description="返回最相似的前 N 个"),
    exclude_object_ids: str = Form("", description="要排除的氚云ObjectId，逗号分隔（排除自身，避免搜到自己）"),
):
    """上传PDF查找相似图纸（BGE图文联合向量 + pgvector搜索）"""
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

    # 提取文本
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

    # BGE 图文联合编码
    query_vectors = np.stack(
        [fe.extract(page.image, text=query_text[:2000]) for page in doc.pages], axis=0
    ).astype(np.float32)

    # 解析 exclude_object_ids → 查 doc_ids
    exclude_doc_ids = None
    if exclude_object_ids.strip():
        h3yun_ids = [x.strip() for x in exclude_object_ids.split(",") if x.strip()]
        if h3yun_ids:
            exclude_doc_ids = await DocStore.find_doc_ids_by_h3yun_ids(h3yun_ids)

    # pgvector 向量搜索（一次搞定，不再需要三段融合）
    hits = await VectorStore.multi_page_search(query_vectors, k=top_k * 2, exclude_doc_ids=exclude_doc_ids)

    # 去重 + 排序
    seen = set()
    results = []
    for hit in hits:
        did = hit["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        target_info = await DocStore.get(did)
        if not target_info:
            continue
        results.append({
            "doc_id": did,
            "score": hit["similarity"],  # BGE向量余弦相似度即最终得分
            "meta": target_info,
        })

    results.sort(key=lambda x: -x["score"])
    final = []
    for i, r in enumerate(results[:top_k]):
        meta = r["meta"]
        final.append(SearchResult(
            doc_id=r["doc_id"],
            score=round(r["score"], 4),
            rank=i + 1,
            image_similarity=round(r["score"], 4),  # BGE统一向量，不再区分
            text_similarity=None,  # 已融合在向量中
            field_similarity=None,  # 已融合在向量中
            meta=_info_out(meta) if meta else None,
        ))

    return SearchResponse(ok=True, query_doc_id=None, num_pages=doc.num_pages, results=final)


@app.post("/api/v1/search/{doc_id}", response_model=SearchResponse, tags=["Search"])
async def search_by_doc_id(doc_id: str, top_k: int = 10):
    """根据库中已有文档 ID 查找相似图纸"""
    info = await DocStore.get(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="doc_id 不存在")

    # 从 PG 获取该文档的向量
    doc_vectors = await VectorStore.get_for_doc(doc_id)
    if doc_vectors.size == 0:
        raise HTTPException(status_code=404, detail="该文档向量不在库中")

    # pgvector 向量搜索（排除自身）
    hits = await VectorStore.multi_page_search(
        doc_vectors, k=top_k * 2, exclude_doc_ids=[doc_id]
    )

    seen = set()
    results = []
    for hit in hits:
        did = hit["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        target_info = await DocStore.get(did)
        if not target_info:
            continue
        results.append({
            "doc_id": did,
            "score": hit["similarity"],
            "meta": target_info,
        })

    results.sort(key=lambda x: -x["score"])
    final = []
    for i, r in enumerate(results[:top_k]):
        final.append(SearchResult(
            doc_id=r["doc_id"],
            score=round(r["score"], 4),
            rank=i + 1,
            image_similarity=round(r["score"], 4),
            text_similarity=None,
            field_similarity=None,
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
    vec_removed = await VectorStore.remove(doc_id)
    await DocStore.remove(doc_id)
    return JSONResponse(
        status_code=200,
        content={"ok": True, "doc_id": doc_id, "vectors_removed": vec_removed},
    )


@app.get("/", tags=["Root"])
def root():
    return {
        "name": "PDF Drawings Similarity API (BGE多模态版)",
        "version": "5.0.0",
        "docs": "/docs",
        "embedding": "Visualized-BGE 768维图文联合嵌入",
        "storage": "PostgreSQL + pgvector",
        "endpoints": {
            "upload": "POST /api/v1/documents",
            "search_by_file": "POST /api/v1/search",
            "search_by_id": "POST /api/v1/search/{doc_id}",
            "list": "GET /api/v1/documents",
            "delete": "DELETE /api/v1/documents/{doc_id}",
            "health": "GET /api/v1/health",
        },
    }
