"""
文档元数据存储（JSON）- 扩展支持 OCR 文本和关键词段
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DocInfo:
    doc_id: str
    filename: str
    size_bytes: int
    num_pages: int
    created_at: float
    phash: str = ""
    signature: str = ""
    ocr_text: str = ""
    text_source: str = "pdf_text"
    ocr_used: bool = False
    # 工程关键字段
    material: str = ""
    process_text: str = ""
    surface_text: str = ""
    tolerance_text: str = ""
    dimension_text: str = ""
    # 额外信息
    extra: dict = field(default_factory=dict)


class DocStore:
    def __init__(self, meta_path: Path):
        self.meta_path = Path(meta_path)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._docs: Dict[str, DocInfo] = {}
        self._load()

    def _load(self):
        if not self.meta_path.exists():
            return
        try:
            raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
            for d in raw:
                info = DocInfo(
                    doc_id=d["doc_id"],
                    filename=d.get("filename", ""),
                    size_bytes=int(d.get("size_bytes", 0)),
                    num_pages=int(d.get("num_pages", 0)),
                    created_at=float(d.get("created_at", time.time())),
                    phash=d.get("phash", ""),
                    signature=d.get("signature", ""),
                    ocr_text=d.get("ocr_text", ""),
                    text_source=d.get("text_source", "pdf_text"),
                    ocr_used=bool(d.get("ocr_used", False)),
                    material=d.get("material", ""),
                    process_text=d.get("process_text", ""),
                    surface_text=d.get("surface_text", ""),
                    tolerance_text=d.get("tolerance_text", ""),
                    dimension_text=d.get("dimension_text", ""),
                    extra=d.get("extra", {}),
                )
                self._docs[info.doc_id] = info
        except Exception:
            self._docs = {}

    def _save(self):
        data = [asdict(v) for v in self._docs.values()]
        self.meta_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:24]

    def add(self, info: DocInfo) -> DocInfo:
        with self._lock:
            self._docs[info.doc_id] = info
            self._save()
            return info

    def get(self, doc_id: str) -> Optional[DocInfo]:
        return self._docs.get(doc_id)

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._docs:
                self._docs.pop(doc_id, None)
                self._save()
                return True
            return False

    def list_all(self, skip: int = 0, limit: int = 100) -> List[DocInfo]:
        items = sorted(self._docs.values(), key=lambda d: -d.created_at)
        return items[skip: skip + limit]

    def find_by_signature(self, sig: str) -> Optional[DocInfo]:
        for d in self._docs.values():
            if d.signature == sig:
                return d
        return None

    def count(self) -> int:
        return len(self._docs)

    def get_meta_dict(self, doc_id: str) -> Dict[str, str]:
        """获取文档的关键字段字典（用于相似度计算）"""
        info = self.get(doc_id)
        if not info:
            return {}
        return {
            "material": info.material,
            "process_text": info.process_text,
            "surface_text": info.surface_text,
            "tolerance_text": info.tolerance_text,
            "dimension_text": info.dimension_text,
        }
