"""
HKBU Course Assistant - Configuration Management
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _parse_bool(value: str, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass
class RAGConfig:
    data_dir: str = "./course_docs"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    chroma_path: str = "./chroma_db"
    rebuild_if_changed: bool = True
    ollama_embed_model: str = "nomic-embed-text"


@dataclass
class OllamaConfig:
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"


@dataclass
class MongoConfig:
    uri: str = ""


@dataclass
class AppConfig:
    rag: RAGConfig
    ollama: OllamaConfig
    mongo: MongoConfig


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    rag_config = RAGConfig(
        data_dir=os.getenv("RAG_DATA_DIR", "./course_docs"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        chroma_path=os.getenv("RAG_CHROMA_PATH", os.getenv("CHROMA_PATH", "./chroma_db")),
        rebuild_if_changed=_parse_bool(os.getenv("RAG_REBUILD_IF_CHANGED", "true"), default=True),
        ollama_embed_model=os.getenv("RAG_OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    ollama_config = OllamaConfig(
        model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    mongo_config = MongoConfig(
        uri=os.getenv("MONGODB_URI", ""),
    )

    return AppConfig(
        rag=rag_config,
        ollama=ollama_config,
        mongo=mongo_config,
    )


# Global config instance
config = load_config()
