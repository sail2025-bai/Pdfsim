"""PDF 图纸相似性查找 - 配置管理（BGE多模态版）

所有配置均可通过 .env 文件持久化，重启不丢失。
优先级: .env 文件 > 环境变量 > 默认值

环境变量命名规则：
  顶层: FEATURE_DIM, USE_GPU 等
  DB前缀: DB_HOST, DB_PORT, DB_PASSWORD 等
  H3YUN前缀: H3YUN_ENGINE_CODE, H3YUN_PRODUCT_SCHEMA_CODE 等
"""
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
MODEL_DIR = BASE_DIR / "models"

# 加载 .env（项目根目录或 app/ 目录均可）
_env_paths = [
    BASE_DIR.parent / ".env",   # Pdfsim/.env
    BASE_DIR / ".env",          # Pdfsim/app/.env
]
for _p in _env_paths:
    if _p.exists():
        load_dotenv(_p)
        break

INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """PostgreSQL / Supabase 数据库配置"""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "postgres"
    max_connections: int = 10
    schema: str = "public"


class H3YunConfig(BaseModel):
    """氚云对接配置"""
    engine_code: str = ""
    engine_secret: str = ""
    base_url: str = "https://www.h3yun.com"
    product_schema_code: str = ""        # 产品表单编码
    attachment_field_code: str = ""       # 附件字段编码
    name_field_code: str = "Name"         # 标识字段编码（产品名称/编号）


class Settings(BaseModel):
    """运行时配置（支持 .env 文件持久化）"""
    image_size: int = 224
    render_dpi: int = 150
    feature_dim: int = 768          # BGE-base 输出维度
    top_k: int = 10
    use_gpu: bool = False
    max_upload_mb: int = 50
    save_files: bool = False
    # BGE 模型配置
    bge_model_name: str = "BAAI/bge-base-en-v1.5"
    bge_visual_weight: str = ""     # 留空则自动下载，或指定本地 .pth 路径
    # 数据库配置（Supabase PG）
    db: DatabaseConfig = DatabaseConfig()
    # 氚云对接配置
    h3yun: H3YunConfig = H3YunConfig()


def _load_settings() -> Settings:
    """从环境变量/.env构建Settings，自动映射前缀"""
    db_kwargs = {}
    for key in ("host", "port", "user", "password", "database", "max_connections", "schema"):
        env_key = f"DB_{key.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            if key == "port" or key == "max_connections":
                db_kwargs[key] = int(val)
            else:
                db_kwargs[key] = val

    h3yun_kwargs = {}
    for key in ("engine_code", "engine_secret", "base_url",
                "product_schema_code", "attachment_field_code", "name_field_code"):
        env_key = f"H3YUN_{key.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            h3yun_kwargs[key] = val

    top_kwargs = {}
    for key in ("feature_dim", "render_dpi", "top_k", "max_upload_mb", "image_size"):
        env_key = key.upper()
        val = os.environ.get(env_key)
        if val is not None:
            top_kwargs[key] = int(val)

    for key in ("bge_model_name", "bge_visual_weight"):
        env_key = key.upper()
        val = os.environ.get(env_key)
        if val is not None:
            top_kwargs[key] = val

    use_gpu = os.environ.get("USE_GPU")
    if use_gpu is not None:
        top_kwargs["use_gpu"] = use_gpu.lower() in ("true", "1", "yes")

    return Settings(
        **top_kwargs,
        db=DatabaseConfig(**db_kwargs) if db_kwargs else DatabaseConfig(),
        h3yun=H3YunConfig(**h3yun_kwargs) if h3yun_kwargs else H3YunConfig(),
    )


settings = _load_settings()
