"""PDF 图纸相似性查找 - 配置管理（pgvector版）"""
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """PostgreSQL / Supabase 数据库配置"""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "postgres"
    max_connections: int = 10
    # Supabase 自托管时 PG 端口通常映射到 5432 或其他
    schema: str = "public"


class Settings(BaseModel):
    """运行时配置"""
    image_size: int = 224
    render_dpi: int = 150
    feature_dim: int = 512
    top_k: int = 5
    use_gpu: bool = False
    max_upload_mb: int = 50
    save_files: bool = False
    # 数据库配置（Supabase PG）
    db: DatabaseConfig = DatabaseConfig()


settings = Settings()
