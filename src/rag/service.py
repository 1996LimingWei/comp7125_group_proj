"""
RAG Service - Retrieval-Augmented Generation for HKBU Course Data
"""
import os
import json
import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGService:
    """RAG Service for HKBU campus knowledge."""

    def __init__(
        self,
        data_dir: str = "./course_docs",
        chroma_path: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="hkbu_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

        # Build knowledge base if empty
        if self.collection.count() == 0:
            logger.info("Building knowledge base...")
            self._build_knowledge_base()
        else:
            logger.info(f"RAG ready: {self.collection.count()} chunks loaded")

    def _load_documents(self) -> List[Dict[str, str]]:
        """Load all TXT files from the data directory."""
        docs = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        docs.append({
                            "file_name": filename,
                            "text": text,
                        })
                    logger.info(f"Loaded: {filename}")
                except Exception as e:
                    logger.error(f"Failed to read {filename}: {e}")
        return docs

    def _chunk_documents(self, docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Split documents into chunks using token-based sliding window."""
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        snippets = []
        chunk_id = 0

        for doc in docs:
            tokens = tokenizer.encode(doc["text"])
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens)

                if len(chunk_text.strip()) < 50:
                    continue

                snippets.append({
                    "file_name": doc["file_name"],
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                })
                chunk_id += 1

        return snippets

    def _build_knowledge_base(self):
        """Build the ChromaDB knowledge base from course documents."""
        # Load documents
        docs = self._load_documents()
        logger.info(f"Loaded {len(docs)} documents")

        # Chunk documents
        snippets = self._chunk_documents(docs)
        logger.info(f"Created {len(snippets)} chunks")

        if not snippets:
            logger.warning("No snippets to index")
            return

        # Generate embeddings
        texts = [s["text"] for s in snippets]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Store in ChromaDB
        self.collection.add(
            documents=texts,
            ids=[str(i) for i in range(len(snippets))],
            metadatas=[{
                "file_name": s["file_name"],
                "chunk_id": s["chunk_id"],
            } for s in snippets],
        )

        logger.info(f"Knowledge base built: {self.collection.count()} chunks")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """Retrieve top-k relevant chunks for the query."""
        if self.collection.count() == 0:
            logger.warning("No chunks in knowledge base")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas"],
        )

        retrieved = []
        for i, doc in enumerate(results["documents"][0]):
            retrieved.append({
                "content": doc,
                "source": results["metadatas"][0][i].get("file_name", "unknown"),
            })

        return retrieved

    def get_context(self, query: str, k: int = 5) -> Optional[str]:
        """Get formatted context string for LLM prompt."""
        chunks = self.retrieve(query, k=k)
        if not chunks:
            return None

        context = "\n\n---\n\n".join(c["content"] for c in chunks)
        return context
