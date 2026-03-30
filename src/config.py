"""
HKBU Course Assistant - Configuration Management
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    chroma_path: str = "/app/chroma_db"


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
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        chroma_path=os.getenv("CHROMA_PATH", "/app/chroma_db"),
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
