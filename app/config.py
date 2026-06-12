"""PDF 图纸相似性查找 - 配置管理"""
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """PostgreSQL 数据库配置"""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "postgres"
    max_connections: int = 10


class Settings(BaseModel):
    """运行时配置"""
    image_size: int = 224
    render_dpi: int = 150
    feature_dim: int = 512
    top_k: int = 5
    use_gpu: bool = False
    max_upload_mb: int = 50
    # 文件保留策略：False 表示只保存向量索引和元数据
    save_files: bool = False
    # 数据库配置（PostgreSQL）
    db: DatabaseConfig = DatabaseConfig()


settings = Settings()
