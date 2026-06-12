"""
客户端示例：演示如何通过 HTTP 调用 PDF 相似性查找 API

用法：
    # 启动服务
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    # 另开终端
    python examples/client.py /path/to/query.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"


def upload(pdf_path: Path):
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/v1/documents",
            files={"file": (pdf_path.name, f, "application/pdf")},
            data={"allow_existing": "true"},
        )
    r.raise_for_status()
    return r.json()


def search_by_file(pdf_path: Path, top_k: int = 5):
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/v1/search",
            files={"file": (pdf_path.name, f, "application/pdf")},
            data={"top_k": str(top_k)},
        )
    r.raise_for_status()
    return r.json()


def list_docs(limit: int = 10):
    r = requests.get(f"{BASE}/api/v1/documents", params={"limit": limit})
    r.raise_for_status()
    return r.json()


def search_by_id(doc_id: str, top_k: int = 5):
    r = requests.post(
        f"{BASE}/api/v1/search/{doc_id}", params={"top_k": top_k}
    )
    r.raise_for_status()
    return r.json()


def health():
    r = requests.get(f"{BASE}/api/v1/health")
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import json

    print("health:", json.dumps(health(), indent=2, ensure_ascii=False))
    if len(sys.argv) < 2:
        print("用法: python examples/client.py /path/to/query.pdf")
        sys.exit(1)
    p = Path(sys.argv[1])
    print("\n上传:", json.dumps(upload(p), indent=2, ensure_ascii=False))
    print("\n查询:", json.dumps(search_by_file(p, top_k=5), indent=2, ensure_ascii=False))
    print("\n文档列表:", json.dumps(list_docs(), indent=2, ensure_ascii=False))
