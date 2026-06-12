"""
融合版测试脚本：验证视觉特征 + OCR + 关键词段的融合评分
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.feature_extractor import (
    FeatureExtractor,
    field_similarity,
    fusion_score,
    text_similarity,
)
from app.index_store import VectorIndex
from app.pdf_reader import render_pdf
from app.storage import DocInfo, DocStore


def _build_pdf(path: Path, kind: str, material: str = "", process: str = ""):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica", 10)
    c.drawString(72, h - 72, f"Test drawing: {kind}")
    c.drawString(72, h - 92, f"Material: {material}")
    c.drawString(72, h - 112, f"Process: {process}")

    if kind.startswith("rect"):
        x0, y0 = 100, 200
        cell = 40
        rows, cols = 6, 8
        for i in range(rows + 1):
            c.line(x0, y0 + i * cell, x0 + cols * cell, y0 + i * cell)
        for j in range(cols + 1):
            c.line(x0 + j * cell, y0, x0 + j * cell, y0 + rows * cell)
        c.rect(w - 220, 60, 180, 80, stroke=1, fill=0)
    elif kind.startswith("circle"):
        cx, cy = w / 2, h / 2
        for r in range(40, 300, 25):
            c.circle(cx, cy, r, stroke=1, fill=0)
        c.line(cx - 320, cy, cx + 320, cy)
        c.line(cx, cy - 320, cx, cy + 320)
    c.showPage()
    c.save()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pdf-sim-test-"))
    print(f"[info] 临时目录: {tmp}")

    # 1. 合成 PDF（带不同材料/工艺标签）
    pdfs = {
        "A1_rect_SUS304_CNC": tmp / "A1_rect_SUS304_CNC.pdf",
        "A2_rect_SUS304_CNC": tmp / "A2_rect_SUS304_CNC.pdf",  # 同类同材料同工艺
        "A3_rect_SUS316_CNC": tmp / "A3_rect_SUS316_CNC.pdf",  # 同类不同材料
        "B1_circle_6061_milling": tmp / "B1_circle_6061_milling.pdf",  # 不同类
    }
    specs = {
        "A1_rect_SUS304_CNC": ("rect", "SUS304", "CNC"),
        "A2_rect_SUS304_CNC": ("rect", "SUS304", "CNC"),
        "A3_rect_SUS316_CNC": ("rect", "SUS316", "CNC"),
        "B1_circle_6061_milling": ("circle", "6061", "milling"),
    }
    for name, p in pdfs.items():
        kind, mat, proc = specs[name]
        _build_pdf(p, kind, material=mat, process=proc)

    # 2. 初始化
    fe = FeatureExtractor()
    index_dir = tmp / "idx"
    idx = VectorIndex(dim=fe.dim, persist_dir=index_dir)
    store = DocStore(index_dir / "docs.json")

    # 3. 入库 A1, A3, B1（留 A2 做查询）
    docs_to_index = ["A1_rect_SUS304_CNC", "A3_rect_SUS316_CNC", "B1_circle_6061_milling"]
    for name in docs_to_index:
        data = pdfs[name].read_bytes()
        doc = render_pdf(data, filename=pdfs[name].name)
        vectors = np.stack([fe.extract_image(p.image) for p in doc.pages], axis=0)
        doc_id = DocStore.new_id()
        idx.add(doc_id, vectors)
        
        # 提取文本和字段
        ocr_text = f"Material: {specs[name][1]}\nProcess: {specs[name][2]}"
        fields = fe.extract_fields(ocr_text)
        
        store.add(DocInfo(
            doc_id=doc_id,
            filename=pdfs[name].name,
            size_bytes=len(data),
            num_pages=doc.num_pages,
            created_at=0,
            ocr_text=ocr_text,
            material=fields.get("material", ""),
            process_text=fields.get("process_text", ""),
        ))
        print(f"  + 入库: {pdfs[name].name} -> {doc_id}")

    print(f"[info] 索引向量数: {idx.count_vectors()}  文档数: {idx.count_docs()}")

    # 4. 查询 A2（期望: A1 > A3 > B1，因为 A1 和 A2 材料工艺都相同）
    qname = "A2_rect_SUS304_CNC"
    qdata = pdfs[qname].read_bytes()
    qdoc = render_pdf(qdata, filename=pdfs[qname].name)
    q_ocr_text = f"Material: {specs[qname][1]}\nProcess: {specs[qname][2]}"
    q_fields = fe.extract_fields(q_ocr_text)
    
    print(f"\n[query] {pdfs[qname].name}")
    print(f"  查询材料: {q_fields.get('material')}")
    print(f"  查询工艺: {q_fields.get('process_text')}")
    
    merged = {}
    for page in qdoc.pages:
        qv = fe.extract_image(page.image)
        hits = idx.search(qv, k=10)
        for h in hits:
            d = h["doc_id"]
            if d not in merged or h["score"] > merged[d]:
                merged[d] = h["score"]

    # 计算融合评分
    results = []
    for did, image_sim in merged.items():
        info = store.get(did)
        if not info:
            continue
        text_sim = text_similarity(q_ocr_text, info.ocr_text or "")
        field_sim = field_similarity(q_fields, store.get_meta_dict(did))
        final_score = fusion_score(image_sim, text_sim, field_sim)
        results.append({
            "name": info.filename,
            "score": final_score,
            "image_sim": image_sim,
            "text_sim": text_sim,
            "field_sim": field_sim,
        })

    results.sort(key=lambda x: -x["score"])
    for i, r in enumerate(results):
        print(f"    rank#{i+1}  score={r['score']:.4f}")
        print(f"        image={r['image_sim']:.4f}  text={r['text_sim']:.4f}  field={r['field_sim']:.4f}")
        print(f"        {r['name']}")

    # 验证断言
    top1 = results[0]
    assert "A1" in top1["name"], f"期望 top-1 是 A1，实际: {top1['name']}"
    print("\n[PASS] 融合评分验证成功！")
    print("A2 (SUS304/CNC) -> A1 (SUS304/CNC) 相似度最高（同图同材料同工艺）")


if __name__ == "__main__":
    main()
