"""
PostgreSQL 存储层单元测试（带 mock）

验证：
  1. DocRecord 序列化/反序列化
  2. VectorStore vec <-> bytes 转换
  3. VectorIndex 的 add / search / remove / rebuild 逻辑
  4. 重启后从 PG 加载恢复
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_doc_record():
    from app.db import DocRecord
    info = DocRecord(
        doc_id="test123",
        filename="test.pdf",
        size_bytes=12345,
        num_pages=5,
        created_at=999.0,
        phash="abc123",
        signature="sig456",
        ocr_text="hello world",
        text_source="pdf_text",
        ocr_used=False,
        material="SUS304",
        process_text="CNC",
        surface_text="阳极氧化",
        tolerance_text="±0.01",
        dimension_text="100x50",
    )
    d = info.to_dict()
    assert d["doc_id"] == "test123"
    assert d["material"] == "SUS304"
    assert d["process_text"] == "CNC"
    assert d["surface_text"] == "阳极氧化"

    row = {
        "doc_id": "test123", "filename": "test.pdf",
        "size_bytes": 12345, "num_pages": 5, "created_at": 999.0,
        "phash": "abc123", "signature": "sig456",
        "ocr_text": "hello world", "text_source": "pdf_text",
        "ocr_used": False,
        "material": "SUS304", "process_text": "CNC",
        "surface_text": "阳极氧化", "tolerance_text": "±0.01",
        "dimension_text": "100x50", "extra": "{}"
    }

    restored = DocRecord.from_row(row)
    assert restored.doc_id == "test123"
    assert restored.material == "SUS304"
    assert restored.ocr_used == False
    print("  ✓ DocRecord 序列化/反序列化 OK")


def test_vector_bytes():
    from app.db import vec_to_bytes, bytes_to_vec

    # 单个向量
    v = np.random.randn(512).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-6)

    # 序列化
    b = vec_to_bytes(v)
    assert isinstance(b, bytes)
    assert len(b) == 512 * 4  # float32 = 4 bytes

    # 反序列化
    v2 = bytes_to_vec(b, dim=512)
    assert v2.shape == (512,)
    assert np.allclose(v, v2, atol=1e-5)
    print("  ✓ 向量 <-> bytes 转换 OK")


def test_vector_index_logic():
    """测试 VectorIndex 的内存逻辑（不依赖真实 PG）"""
    import faiss
    from app.index_store import VectorIndex

    with tempfile.TemporaryDirectory() as tmp:
        # Mock PG 层
        all_vectors = []
        all_doc_ids = []
        doc2fids_ref = {}

        class MockVectorStore:
            def __init__(self):
                self._storage = {}  # doc_id -> list of vectors

            async def add(self, doc_id, vectors):
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                self._storage.setdefault(doc_id, []).extend([v.copy() for v in vectors])
                return list(range(len(self._storage[doc_id])))

            async def remove(self, doc_id):
                self._storage.pop(doc_id, None)
                return 1

            async def get_all(self):
                if not self._storage:
                    return np.zeros((0, 512), dtype=np.float32), [], []
                all_vecs, all_ids = [], []
                for did in sorted(self._storage.keys()):
                    for vec in self._storage[did]:
                        all_vecs.append(vec)
                        all_ids.append(did)
                return np.stack(all_vecs, axis=0).astype(np.float32), all_ids, list(range(len(all_vecs)))

            async def count(self):
                return sum(len(v) for v in self._storage.values())

        # 创建索引（用 mock PG）
        with patch("app.index_store._get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_vs = MockVectorStore()
            mock_db.VectorStore = mock_vs
            mock_get_db.return_value = mock_db

            idx = VectorIndex(dim=512)
            loop = asyncio.new_event_loop()

            # 1. 添加 3 个文档
            doc_ids = []
            for i in range(3):
                n_pages = [2, 1, 3][i]
                v = np.random.randn(n_pages, 512).astype(np.float32)
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                did = f"doc_{i}"
                doc_ids.append(did)
                loop.run_until_complete(idx.add(did, v))

            assert idx.count_vectors() == 6
            assert idx.count_docs() == 3
            print(f"  ✓ 添加 3 个文档 (共6页向量) OK")

            # 2. 搜索
            q = np.random.randn(512).astype(np.float32)
            q /= np.linalg.norm(q)
            hits = idx.search(q, k=3)
            assert len(hits) == 3
            assert all("doc_id" in h and "score" in h for h in hits)
            print(f"  ✓ 搜索返回 {len(hits)} 个结果 OK")

            # 3. 删除一个文档
            removed = loop.run_until_complete(idx.remove_doc("doc_1"))
            assert idx.count_docs() == 2
            assert idx.count_vectors() == 5
            print(f"  ✓ 删除 doc_1 成功 OK")

            # 4. 重建索引后数据一致
            loop.run_until_complete(idx.load_from_pg())
            assert idx.count_docs() == 2
            assert idx.count_vectors() == 5
            print(f"  ✓ 从 PG 重建索引 OK")

            loop.close()

    print("  ✓ VectorIndex 内存逻辑 OK")


def test_fusion_scoring():
    """测试融合评分（不依赖 PG）"""
    from app.feature_extractor import fusion_score, text_similarity, field_similarity

    # 两份图纸的相似度
    img_sim = 0.95   # 图像非常相似
    txt_sim = 0.80   # OCR 文本相似
    fld_sim = 0.70   # 字段有差异

    score = fusion_score(img_sim, txt_sim, fld_sim, 0.70, 0.20, 0.10)
    expected = 0.70 * 0.95 + 0.20 * 0.80 + 0.10 * 0.70
    assert abs(score - expected) < 1e-6, f"融合评分错误: {score} vs {expected}"

    # 材料不同，字段分数拉低
    fld_sim2 = 0.0
    score2 = fusion_score(img_sim, txt_sim, fld_sim2, 0.70, 0.20, 0.10)
    expected2 = 0.70 * 0.95 + 0.20 * 0.80 + 0.10 * 0.0
    assert abs(score2 - expected2) < 1e-6
    print(f"  ✓ 融合评分: score1={score:.4f} score2={score2:.4f} OK")


def test_text_similarity():
    from app.feature_extractor import text_similarity
    s = text_similarity(
        "Material: SUS304 Process: CNC",
        "Material: SUS304 Process: CNC"
    )
    assert s > 0.95, f"相同文本相似度应很高: {s}"
    s2 = text_similarity(
        "Material: SUS304 Process: CNC",
        "Material: 6061 Process: milling"
    )
    assert s2 < 0.7, f"不同文本相似度应较低: {s2}"
    print(f"  ✓ 文本相似度: 相同={s:.4f} 不同={s2:.4f} OK")


def main():
    print("[test_db] 开始 PG 存储层单元测试\n")

    print("=== DocRecord ===")
    test_doc_record()

    print("\n=== 向量序列化 ===")
    test_vector_bytes()

    print("\n=== 融合评分 ===")
    test_fusion_scoring()
    test_text_similarity()

    print("\n=== VectorIndex 逻辑（mock PG）===")
    test_vector_index_logic()

    print("\n[PASS] 所有测试通过！")
    print("\n结论：")
    print("  - PG 存储层的数据模型和序列化逻辑验证无误")
    print("  - FAISS + PG 协同工作的索引逻辑已验证")
    print("  - 融合评分算法验证无误")


if __name__ == "__main__":
    main()
