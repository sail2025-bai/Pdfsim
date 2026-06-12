"""
测试脚本：验证 PDF 相似性查找核心链路
不依赖 FastAPI，直接调用底层模块（离线快速验证）。

运行：
    pip install -r requirements.txt
    python tests/test_pipeline.py

说明：
  程序会在临时目录中生成几张合成 PDF（模拟工程图纸）：
    A1.pdf / A2.pdf  (同一类：矩形为主)
    B1.pdf          (另一类：圆形为主)
  入库 A1 / B1，用 A2 查询 -> A2 应命中 A1 为 top-1。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.feature_extractor import FeatureExtractor
from app.index_store import VectorIndex
from app.pdf_reader import render_pdf
from app.storage import DocInfo, DocStore


# ---------- 合成 PDF（用 ReportLab） ----------
def _build_pdf(path: Path, kind: str):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    # 加一些文字元信息（不影响视觉相似度）
    c.setFont("Helvetica", 10)
    c.drawString(72, h - 72, f"Test drawing: {kind}")
    c.drawString(72, h - 92, "Dwg-No: 2024-001")

    if kind.startswith("rect"):
        # 矩形网格
        x0, y0 = 100, 200
        cell = 40
        rows, cols = 6, 8
        for i in range(rows + 1):
            c.line(x0, y0 + i * cell, x0 + cols * cell, y0 + i * cell)
        for j in range(cols + 1):
            c.line(x0 + j * cell, y0, x0 + j * cell, y0 + rows * cell)
        # 标题框
        c.rect(w - 220, 60, 180, 80, stroke=1, fill=0)
    elif kind.startswith("circle"):
        # 一堆同心圆
        cx, cy = w / 2, h / 2
        for r in range(40, 300, 25):
            c.circle(cx, cy, r, stroke=1, fill=0)
        c.line(cx - 320, cy, cx + 320, cy)
        c.line(cx, cy - 320, cx, cy + 320)
    elif kind.startswith("diag"):
        # 斜线
        x0, y0 = 100, 200
        for i in range(15):
            c.line(x0, y0 + i * 20, x0 + 600, y0 + i * 20 + 150)
    else:
        c.rect(100, 100, 400, 400)
    c.showPage()
    c.save()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pdf-sim-test-"))
    print(f"[info] 临时目录: {tmp}")

    # 1. 合成 PDF
    pdfs = {
        "A1_rect": tmp / "A1_rect.pdf",
        "A2_rect_variant": tmp / "A2_rect_variant.pdf",
        "B1_circle": tmp / "B1_circle.pdf",
        "C1_diag": tmp / "C1_diag.pdf",
    }
    for name, p in pdfs.items():
        _build_pdf(p, name.split("_")[1] if "_" in name else name)

    # 2. 初始化
    fe = FeatureExtractor()
    index_dir = tmp / "idx"
    idx = VectorIndex(dim=fe.dim, persist_dir=index_dir)
    store = DocStore(index_dir / "docs.json")

    # 3. 入库（A1, B1, C1）；留 A2 做查询
    docs_to_index = ["A1_rect", "B1_circle", "C1_diag"]
    for name in docs_to_index:
        data = pdfs[name].read_bytes()
        doc = render_pdf(data, filename=pdfs[name].name)
        vectors = np.stack([fe.extract(p.image) for p in doc.pages], axis=0)
        doc_id = DocStore.new_id()
        idx.add(doc_id, vectors)
        store.add(DocInfo(
            doc_id=doc_id,
            filename=pdfs[name].name,
            size_bytes=len(data),
            num_pages=doc.num_pages,
            created_at=0,
        ))
        print(f"  + 入库: {pdfs[name].name} -> {doc_id}  num_pages={doc.num_pages}")

    print(f"[info] 索引向量数: {idx.count_vectors()}  文档数: {idx.count_docs()}")

    # 4. 查询 A2（与 A1 同类），期望 top-1 是 A1
    qname = "A2_rect_variant"
    qdata = pdfs[qname].read_bytes()
    qdoc = render_pdf(qdata, filename=pdfs[qname].name)
    print(f"\n[query] {pdfs[qname].name}:")
    merged = {}
    for page in qdoc.pages:
        qv = fe.extract(page.image)
        hits = idx.search(qv, k=3)
        for h in hits:
            d = h["doc_id"]
            if d not in merged or h["score"] > merged[d]:
                merged[d] = h["score"]
    ranked = sorted(merged.items(), key=lambda x: -x[1])
    for i, (did, s) in enumerate(ranked):
        info = store.get(did)
        print(f"    rank#{i+1}  score={s:.4f}  {info.filename if info else did}")

    # 基本断言
    top1_doc = store.get(ranked[0][0])
    assert top1_doc is not None and "A1" in top1_doc.filename, (
        f"期望 top-1 是 A1，实际: {top1_doc.filename if top1_doc else None}"
    )
    print("\n[PASS] 相似性检索成功！")

    # 5. 删除 + 持久化测试
    test_doc_id = store.list_all()[0].doc_id
    before = idx.count_vectors()
    idx.remove_doc(test_doc_id)
    idx.save()
    print(f"[info] 删除后索引向量数: {idx.count_vectors()} (之前 {before})")
    idx2 = VectorIndex(dim=fe.dim, persist_dir=index_dir)
    idx2.load()
    assert idx2.count_vectors() == idx.count_vectors(), "持久化/加载不匹配"
    print("[PASS] 持久化 / 删除测试通过")


if __name__ == "__main__":
    main()
