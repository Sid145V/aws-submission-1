import os
from pathlib import Path
from dotenv import load_dotenv

# Base project path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Config:
    """
    RAG Assistant Configuration loaded from environment variables.
    """
    # Gemini API Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    # RAG Settings
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Database Settings
    _db_url = os.getenv("DATABASE_URL", "sqlite:///logs/rag_system.db")
    if _db_url.startswith("sqlite:///"):
        _db_path = _db_url.replace("sqlite:///", "")
        if not os.path.isabs(_db_path):
            _db_url = f"sqlite:///{(BASE_DIR / _db_path).as_posix()}"
    DATABASE_URL: str = _db_url

    # Paths
    PROJECT_ROOT: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Server Settings
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    FRONTEND_PORT: int = int(os.getenv("FRONTEND_PORT", "8501"))

    # Ensure crucial directories exist
    @classmethod
    def initialize_directories(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize paths on config import
Config.initialize_directories()
