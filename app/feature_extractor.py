"""
特征提取模块（融合版）

核心设计：
  - 图像特征：pHash + 分块pHash + Sobel梯度直方图（无torch时）或 ResNet-50（有torch时）
  - OCR文本：PaddleOCR（可选，无则用PDF原生文本）
  - 关键词段：材料、工艺、表面处理、公差、尺寸等工程语义信息

评分权重：
  图像结构：70%
  OCR全文：20%
  关键字段：10%
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
    _HAS_PADDLE = True
except Exception:
    PaddleOCR = None
    _HAS_PADDLE = False

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import models as _tm
    _HAS_TORCH = True
    _torch_no_grad = torch.no_grad
except Exception:
    torch = None
    nn = None
    T = None
    _tm = None
    _torch_no_grad = None

# ====================
# 工程关键词配置
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

# ====================
# OCR引擎
# ====================
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        if not _HAS_PADDLE:
            raise RuntimeError("未安装 PaddleOCR：pip install paddleocr paddlepaddle")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return _ocr_engine

# ====================
# 图像预处理（借鉴用户方案）
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

def _preprocess_for_feature(img_arr):
    """完整的图纸图像预处理流水线"""
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    gray = _crop_border(gray)
    gray = _deskew(gray)
    gray = _normalize_size(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

# ====================
# pHash特征
# ====================
def phash_bits(pil_img: Image.Image, hash_size: int = 8) -> np.ndarray:
    gray = pil_img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(arr)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low.flatten()[1:]) if dct_low.size > 1 else 0.0
    return (dct_low > med).astype(np.float32).flatten()

def phash_hex(pil_img: Image.Image, hash_size: int = 8) -> str:
    bits = phash_bits(pil_img, hash_size)
    n = int("".join(str(int(b)) for b in bits), 2)
    width = hash_size * hash_size // 4
    return f"{n:0{width}x}"

def _subregion_phash(pil_img: Image.Image, grid: int = 3, hash_size: int = 6) -> np.ndarray:
    w, h = pil_img.size
    block_w, block_h = w // grid, h // grid
    gray = pil_img.convert("L")
    feats = []
    for i in range(grid):
        for j in range(grid):
            region = gray.crop((j * block_w, i * block_h,
                               (j + 1) * block_w if j < grid - 1 else w,
                               (i + 1) * block_h if i < grid - 1 else h))
            feats.append(phash_bits(region, hash_size=hash_size))
    return np.concatenate(feats, axis=0)

def _hog_features(pil_img: Image.Image, bins: int = 36) -> np.ndarray:
    gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) * 180.0 / np.pi
    mask = mag > (mag.mean() + 0.3 * mag.std())
    sel_ang = ang[mask]
    if sel_ang.size < 10:
        return np.zeros(bins, dtype=np.float32)
    h, _ = np.histogram(sel_ang, bins=bins, range=(-180, 180))
    h = h.astype(np.float32)
    h /= (np.linalg.norm(h) + 1e-6)
    return h

# ====================
# 深度特征（可选）
# ====================
class _DeepFeatureExtractor:
    def __init__(self, deep_dim: int = 256):
        weights = _tm.ResNet50_Weights.IMAGENET1K_V2
        backbone = _tm.resnet50(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )
        self.proj = nn.Linear(2048, deep_dim)
        nn.init.eye_(self.proj.weight[:, :deep_dim])
        nn.init.zeros_(self.proj.bias)
        self.features.eval()
        self.proj.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, pil_img: Image.Image) -> np.ndarray:
        with _torch_no_grad():
            x = self.transform(pil_img.convert("RGB")).unsqueeze(0)
            feat = self.features(x)
            feat = self.proj(feat)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)

# ====================
# 关键词段提取
# ====================
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
    """提取材料、工艺、表面处理、公差、尺寸等关键字段"""
    return {
        "material": ",".join(_find_keywords(text, MATERIAL_KEYWORDS)),
        "process_text": ",".join(_find_keywords(text, PROCESS_KEYWORDS)),
        "surface_text": ",".join(_find_keywords(text, SURFACE_KEYWORDS)),
        "tolerance_text": ",".join(_extract_tolerance(text)),
        "dimension_text": ",".join(_extract_dimensions(text))
    }

# ====================
# 文本相似度计算
# ====================
try:
    from rapidfuzz import fuzz
    def text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return fuzz.token_set_ratio(a, b) / 100.0
except Exception:
    def text_similarity(a: str, b: str) -> float:
        """fallback：简单的Jaccard相似度"""
        if not a or not b:
            return 0.0
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

def field_similarity(query_meta: dict, target_meta: dict) -> float:
    """关键字段相似度（加权）"""
    weights = {
        "material": 0.35,
        "process_text": 0.25,
        "surface_text": 0.20,
        "tolerance_text": 0.10,
        "dimension_text": 0.10
    }
    total = 0.0
    score = 0.0
    for key, weight in weights.items():
        q = query_meta.get(key, "")
        t = target_meta.get(key, "")
        if q:
            total += weight
            if t:
                score += weight * text_similarity(q, t)
    return score / total if total > 0 else 0.0

# ====================
# 统一提取器
# ====================
class FeatureExtractor:
    """融合视觉特征 + OCR + 关键词段的统一提取器"""
    
    def __init__(self):
        self._deep = None
        if _HAS_TORCH:
            try:
                self._deep = _DeepFeatureExtractor(deep_dim=256)
            except Exception:
                self._deep = None
        
        self._phash_grid = 3
        self._phash_size = 6
        self._edge_bins = 128
        self._dim = 512
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def extract_image(self, pil_img: Image.Image) -> np.ndarray:
        """提取图像视觉特征（512维向量）"""
        if self._deep is not None:
            return self._extract_with_deep(pil_img)
        return self._extract_traditional(pil_img)
    
    def _extract_traditional(self, pil_img: Image.Image) -> np.ndarray:
        g_phash = phash_bits(pil_img, hash_size=8)  # 64
        s_phash = _subregion_phash(pil_img, grid=3, hash_size=6)  # 324
        edge = _hog_features(pil_img, bins=self._edge_bins)  # 128
        
        g_phash = g_phash / (np.linalg.norm(g_phash) + 1e-6)
        s_phash = s_phash / (np.linalg.norm(s_phash) + 1e-6)
        edge = edge / (np.linalg.norm(edge) + 1e-6)
        
        vec = np.concatenate([g_phash, s_phash, edge], axis=0).astype(np.float32)
        vec = vec[:self._dim]
        if vec.size < self._dim:
            pad = np.zeros(self._dim - vec.size, dtype=np.float32)
            vec = np.concatenate([vec, pad], axis=0)
        vec /= (np.linalg.norm(vec) + 1e-6)
        return vec
    
    def _extract_with_deep(self, pil_img: Image.Image) -> np.ndarray:
        deep = self._deep.forward(pil_img)  # 256
        g_phash = phash_bits(pil_img, hash_size=8)  # 64
        edge = _hog_features(pil_img, bins=192)  # 192
        
        g_phash = g_phash / (np.linalg.norm(g_phash) + 1e-6)
        edge = edge / (np.linalg.norm(edge) + 1e-6)
        
        vec = np.concatenate([deep, g_phash, edge], axis=0).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-6)
        return vec
    
    def extract_ocr(self, img_arr: np.ndarray) -> str:
        """从图像提取OCR文本（若无PaddleOCR则返回空字符串）"""
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
# 融合评分（图像70% + 文本20% + 字段10%）
# ====================
def fusion_score(
    image_sim: float,
    text_sim: float,
    field_sim: float,
    weight_image: float = 0.70,
    weight_text: float = 0.20,
    weight_field: float = 0.10
) -> float:
    """计算融合相似度得分"""
    return (
        weight_image * image_sim +
        weight_text * text_sim +
        weight_field * field_sim
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))
