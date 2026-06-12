"""
PDF 预处理模块
- 将 PDF 渲染为图像
- 提取文本元数据
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


@dataclass
class PdfPage:
    page_index: int
    width: int
    height: int
    image: Image.Image  # PIL RGB 图像
    text: str


@dataclass
class PdfDoc:
    filename: str
    num_pages: int
    pages: List[PdfPage]


def _pil_from_bytes(pix) -> Image.Image:
    """fitz.Pixmap -> PIL.Image"""
    if pix.n - pix.alpha >= 4:  # CMYK
        pix = fitz.Pixmap(fitz.csRGB, pix)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img


def render_pdf(
    pdf_bytes: bytes,
    filename: str = "",
    dpi: int = 150,
    max_pages: Optional[int] = None,
) -> PdfDoc:
    """
    将 PDF 渲染为图像。
    图纸类 PDF 通常为矢量，渲染为高分辨率图像后做视觉相似度效果最好。
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: List[PdfPage] = []
    try:
        total = doc.page_count
        limit = min(total, max_pages) if max_pages else total
        for i in range(limit):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = _pil_from_bytes(pix)
            text = page.get_text("text") or ""
            pages.append(
                PdfPage(
                    page_index=i,
                    width=img.width,
                    height=img.height,
                    image=img,
                    text=text.strip(),
                )
            )
    finally:
        doc.close()
    return PdfDoc(filename=filename, num_pages=len(pages), pages=pages)


def render_first_page(pdf_bytes: bytes, dpi: int = 200) -> Image.Image:
    """快速只渲染第一页，用于搜索场景。"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(0)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return _pil_from_bytes(pix)
    finally:
        doc.close()


def thumbnail(pil_img: Image.Image, size=(256, 256)) -> bytes:
    """生成缩略图 PNG，便于前端展示。"""
    img = pil_img.copy()
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_numpy_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)
