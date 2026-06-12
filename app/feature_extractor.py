"""
特征提取模块（BGE多模态版）

核心设计：
  - 图像+OCR文本 → Visualized-BGE 统一编码为 768维向量
  - 图文联合嵌入：model.encode(image=图纸, text=OCR文本)
  - 无需手工设计权重，向量空间自动对齐视觉和语义

优势 vs 旧版(pHash+HOG)：
  - 768维语义密集向量 vs 512维稀疏二值向量
  - 一次向量搜索 vs 三段融合评分
  - 深度语义理解 vs 浅层像素特征
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ====================
# Visualized-BGE 加载
# ====================
_BGE_MODEL = None

def _get_bge_model():
    """懒加载 Visualized-BGE 模型（首次调用时加载，之后复用）"""
    global _BGE_MODEL
    if _BGE_MODEL is not None:
        return _BGE_MODEL

    try:
        from visual_bge.modeling import Visualized_BGE
        from app.config import settings

        model_name = settings.bge_model_name
        model_weight = settings.bge_visual_weight or None

        logger.info(f"Loading Visualized-BGE: {model_name}, weight={model_weight or 'auto-download'}")
        _BGE_MODEL = Visualized_BGE(
            model_name_bge=model_name,
            model_weight=model_weight,
        )
        _BGE_MODEL.eval()
        logger.info("Visualized-BGE loaded successfully (CPU mode)")
        return _BGE_MODEL

    except ImportError as e:
        logger.error(f"visual_bge not installed: {e}")
        logger.error("Install: pip install FlagEmbedding && install visual_bge submodule")
        raise
    except Exception as e:
        logger.error(f"Failed to load Visualized-BGE: {e}")
        raise


# ====================
# OCR引擎（PaddleOCR，可选）
# ====================
_HAS_PADDLE = False
try:
    from paddleocr import PaddleOCR
    _HAS_PADDLE = True
except Exception:
    PaddleOCR = None

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        if not _HAS_PADDLE:
            raise RuntimeError("未安装 PaddleOCR：pip install paddleocr paddlepaddle")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return _ocr_engine


# ====================
# 图像预处理
# ====================
def _crop_border(gray):
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(255 - th)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    if w < 100 or h < 100:
        return gray
    return gray[y:y+h, x:x+w]

def _deskew(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return gray
    angles = []
    for line in lines[:50]:
        _, theta = line[0]
        angle = theta * 180 / np.pi
        if angle < 45:
            angles.append(angle)
        elif angle > 135:
            angles.append(angle - 180)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    if abs(median_angle) > 5:
        return gray
    h, w = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _normalize_size(gray, max_side=1600):
    h, w = gray.shape
    scale = max_side / max(h, w)
    if scale >= 1:
        return gray
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def preprocess_for_ocr(img_arr: np.ndarray) -> np.ndarray:
    """预处理图纸图像（裁边+纠偏+归一化+增强+二值化）"""
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    gray = _crop_border(gray)
    gray = _deskew(gray)
    gray = _normalize_size(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


# ====================
# 工程关键词提取（保留用于元数据字段，不参与向量）
# ====================
MATERIAL_KEYWORDS = [
    "SUS304", "SUS316", "316L", "304", "316",
    "Q235", "Q345", "45#", "45钢", "40Cr",
    "6061", "6063", "7075", "5052",
    "铝合金", "不锈钢", "碳钢", "铜", "黄铜",
    "POM", "尼龙", "ABS", "PC", "PP", "PE"
]

PROCESS_KEYWORDS = [
    "车削", "铣削", "钻孔", "攻牙", "攻丝", "倒角", "去毛刺",
    "线切割", "激光切割", "折弯", "焊接", "磨削",
    "CNC", "加工中心", "数控车", "镗孔", "铰孔", "沉孔"
]

SURFACE_KEYWORDS = [
    "阳极氧化", "硬质氧化", "发黑", "镀锌", "镀镍", "镀铬",
    "喷砂", "喷涂", "烤漆", "抛光", "拉丝", "电泳",
    "氧化", "表面处理", "钝化"
]

import re

def _find_keywords(text: str, keywords: list) -> List[str]:
    text = (text or "").replace(" ", "").upper()
    found = []
    for kw in keywords:
        if kw.upper() in text:
            found.append(kw)
    return sorted(set(found))

def _extract_tolerance(text: str) -> List[str]:
    patterns = [
        r"±\s?\d+\.?\d*", r"\+\s?0\.\d+", r"-\s?0\.\d+",
        r"公差.{0,20}", r"未注公差.{0,30}", r"GB/T\s?\d+\.?\d*"
    ]
    results = []
    for p in patterns:
        results.extend(re.findall(p, text or "", flags=re.IGNORECASE))
    return sorted(set(results))

def _extract_dimensions(text: str) -> List[str]:
    patterns = [
        r"\d+\.?\d*\s?[xX×]\s?\d+\.?\d*(?:\s?[xX×]\s?\d+\.?\d*)?",
        r"Φ\s?\d+\.?\d*", r"φ\s?\d+\.?\d*", r"M\d+"
    ]
    results = []
    for p in patterns:
        results.extend(re.findall(p, text or ""))
    return sorted(set(results))[:50]

def extract_key_fields(text: str) -> Dict[str, str]:
    """提取材料、工艺、表面处理、公差、尺寸等关键字段（元数据，非向量）"""
    return {
        "material": ",".join(_find_keywords(text, MATERIAL_KEYWORDS)),
        "process_text": ",".join(_find_keywords(text, PROCESS_KEYWORDS)),
        "surface_text": ",".join(_find_keywords(text, SURFACE_KEYWORDS)),
        "tolerance_text": ",".join(_extract_tolerance(text)),
        "dimension_text": ",".join(_extract_dimensions(text))
    }


# ====================
# 统一特征提取器（BGE版）
# ====================
class FeatureExtractor:
    """BGE多模态特征提取器

    核心方法：
    - extract(image, text) → 768维图文联合向量
    - extract_image(image) → 768维纯图像向量
    - extract_ocr(img_arr) → OCR文本
    - extract_fields(text) → 工程关键字段
    """

    def __init__(self):
        self._dim = 768

    @property
    def dim(self) -> int:
        return self._dim

    def extract(self, pil_img: Image.Image, text: str = "") -> np.ndarray:
        """提取图文联合嵌入向量（768维）

        Args:
            pil_img: 图纸图像
            text: OCR提取的文本（可选，有则图文联合编码，无则纯图像编码）

        Returns:
            768维 float32 归一化向量
        """
        import torch
        model = _get_bge_model()

        with torch.no_grad():
            if text and text.strip():
                # 图文联合编码：同时捕捉视觉结构和文字语义
                vec = model.encode(image=pil_img, text=text)
            else:
                # 纯图像编码
                vec = model.encode(image=pil_img)

        result = vec.cpu().numpy().flatten().astype(np.float32)
        # 归一化（BGE输出通常已归一化，保险起见再归一化一次）
        norm = np.linalg.norm(result)
        if norm > 1e-6:
            result /= norm
        return result

    def extract_image(self, pil_img: Image.Image) -> np.ndarray:
        """纯图像嵌入（768维）"""
        return self.extract(pil_img, text="")

    def extract_ocr(self, img_arr: np.ndarray) -> str:
        """从图像提取OCR文本"""
        if not _HAS_PADDLE:
            return ""
        try:
            engine = get_ocr_engine()
            result = engine.ocr(img_arr, cls=True)
            texts = []
            if result:
                for page in result:
                    if page:
                        for line in page:
                            try:
                                text = line[1][0]
                                if text:
                                    texts.append(text)
                            except Exception:
                                continue
            return "\n".join(texts)
        except Exception:
            return ""

    def extract_fields(self, text: str) -> Dict[str, str]:
        """从文本提取工程关键字段"""
        return extract_key_fields(text)

    def file_signature(self, pil_img: Image.Image) -> str:
        """基于图像内容的稳定哈希（用于去重）"""
        small = pil_img.convert("RGB").resize((128, 128))
        data = np.asarray(small, dtype=np.uint8).tobytes()
        return hashlib.sha256(data).hexdigest()


# ====================
# 相似度计算（简化版，直接余弦相似度）
# ====================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))
