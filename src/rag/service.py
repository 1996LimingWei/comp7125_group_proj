"""
RAG Service - Retrieval-Augmented Generation for HKBU Course Data
"""
import os
import json
import logging
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Callable, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "hkbu_knowledge"
MANIFEST_ID = "__manifest__"


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    chunk_id: Optional[int]
    start_token: Optional[int]
    end_token: Optional[int]
    distance: Optional[float]


def format_chunks_for_prompt(chunks: Sequence[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        header = f"Source: {c.source}#chunk:{c.chunk_id} distance:{c.distance}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


class RAGService:
    """RAG Service for HKBU campus knowledge."""

    def __init__(
        self,
        data_dir: str = "./course_docs",
        chroma_path: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        rebuild_if_changed: bool = True,
        embedding_model: Optional[Any] = None,
        context_formatter: Optional[Callable[[Sequence[RetrievedChunk]], str]] = None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rebuild_if_changed = rebuild_if_changed
        self._tokenizer = None
        self._context_formatter = context_formatter or format_chunks_for_prompt

        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = embedding_model or SentenceTransformer(self.embedding_model_name)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if self.rebuild_if_changed:
            self._ensure_knowledge_base()
        else:
            if self.collection.count() == 0:
                logger.info("Building knowledge base...")
                self._build_knowledge_base()
            else:
                logger.info(f"RAG ready: {self.collection.count()} chunks loaded")

    def _ensure_knowledge_base(self):
        current_manifest = self._compute_manifest()
        stored_manifest = self._get_stored_manifest()

        if self.collection.count() == 0:
            logger.info("Building knowledge base...")
            self._build_knowledge_base(manifest_text=current_manifest)
            return

        if stored_manifest != current_manifest:
            logger.info("Knowledge base out of date, rebuilding...")
            self._recreate_collection()
            self._build_knowledge_base(manifest_text=current_manifest)
            return

        logger.info(f"RAG ready: {self.collection.count()} chunks loaded")

    def _recreate_collection(self):
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_stored_manifest(self) -> Optional[str]:
        try:
            data = self.collection.get(ids=[MANIFEST_ID], include=["documents"])
            docs = data.get("documents") or []
            if docs and docs[0]:
                return docs[0]
            return None
        except Exception:
            return None

    def _compute_manifest(self) -> str:
        items = []
        try:
            filenames = sorted(
                f for f in os.listdir(self.data_dir)
                if f.endswith(".txt")
            )
        except Exception:
            filenames = []

        for filename in filenames:
            file_path = os.path.join(self.data_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                digest = hashlib.sha256(content).hexdigest()
                items.append({"file_name": filename, "sha256": digest})
            except Exception:
                items.append({"file_name": filename, "sha256": None})

        manifest = {
            "data_dir": os.path.abspath(self.data_dir),
            "files": items,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model_name,
        }
        return json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    def _load_documents(self) -> List[Dict[str, str]]:
        """Load all TXT files from the data directory."""
        docs = []
        try:
            filenames = sorted(os.listdir(self.data_dir))
        except Exception as e:
            logger.error(f"Failed to list data directory {self.data_dir}: {e}")
            return []

        for filename in filenames:
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

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import GPT2TokenizerFast as GPT2Tokenizer
        except Exception:
            try:
                from transformers import GPT2Tokenizer
            except Exception:
                self._tokenizer = None
                return None

        try:
            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            return self._tokenizer
        except Exception:
            self._tokenizer = None
            return None

    def _chunk_documents(self, docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Split documents into chunks using token-based sliding window."""
        tokenizer = self._get_tokenizer()
        snippets = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for doc in docs:
            chunk_index = 0
            if tokenizer is not None:
                tokens = tokenizer.encode(doc["text"])
                for i in range(0, len(tokens), step):
                    chunk_tokens = tokens[i:i + self.chunk_size]
                    chunk_text = tokenizer.decode(chunk_tokens)

                    if len(chunk_text.strip()) < 50:
                        continue

                    chunk_id = hashlib.sha256(
                        f"{doc['file_name']}:{i}:{i + len(chunk_tokens)}:{self.chunk_size}:{self.chunk_overlap}".encode("utf-8")
                    ).hexdigest()

                    snippets.append({
                        "id": chunk_id,
                        "file_name": doc["file_name"],
                        "chunk_id": chunk_index,
                        "start_token": i,
                        "end_token": i + len(chunk_tokens),
                        "text": chunk_text.strip(),
                    })
                    chunk_index += 1
                continue

            words = doc["text"].split()
            for i in range(0, len(words), step):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)

                if len(chunk_text.strip()) < 50:
                    continue

                chunk_id = hashlib.sha256(
                    f"{doc['file_name']}:{i}:{i + len(chunk_words)}:{self.chunk_size}:{self.chunk_overlap}".encode("utf-8")
                ).hexdigest()

                snippets.append({
                    "id": chunk_id,
                    "file_name": doc["file_name"],
                    "chunk_id": chunk_index,
                    "start_token": i,
                    "end_token": i + len(chunk_words),
                    "text": chunk_text.strip(),
                })
                chunk_index += 1

        return snippets

    def _build_knowledge_base(self, manifest_text: Optional[str] = None):
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
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()

        # Store in ChromaDB
        self.collection.add(
            documents=texts,
            ids=[s["id"] for s in snippets],
            metadatas=[{
                "doc_type": "chunk",
                "file_name": s["file_name"],
                "chunk_id": s["chunk_id"],
                "start_token": s["start_token"],
                "end_token": s["end_token"],
            } for s in snippets],
            embeddings=embeddings,
        )

        if manifest_text is None:
            manifest_text = self._compute_manifest()

        manifest_embedding = self.embedding_model.encode([manifest_text]).tolist()
        self.collection.add(
            documents=[manifest_text],
            ids=[MANIFEST_ID],
            metadatas=[{"doc_type": "manifest"}],
            embeddings=manifest_embedding,
        )

        logger.info(f"Knowledge base built: {len(snippets)} chunks indexed")

    def retrieve_chunks(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
            where={"doc_type": "chunk"},
        )

        retrieved: List[RetrievedChunk] = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            if meta.get("doc_type", "chunk") != "chunk":
                continue
            distance = dists[i] if i < len(dists) else None
            retrieved.append(RetrievedChunk(
                content=doc,
                source=meta.get("file_name", "unknown"),
                chunk_id=meta.get("chunk_id"),
                start_token=meta.get("start_token"),
                end_token=meta.get("end_token"),
                distance=distance,
            ))

        return retrieved

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """Retrieve top-k relevant chunks for the query."""
        chunks = self.retrieve_chunks(query, k=k)
        if not chunks:
            if self.collection.count() == 0:
                logger.warning("No chunks in knowledge base")
            return []

        return [
            {
                "content": c.content,
                "source": c.source,
                "chunk_id": c.chunk_id,
                "start_token": c.start_token,
                "end_token": c.end_token,
                "distance": c.distance,
            }
            for c in chunks
        ]

    def get_context(self, query: str, k: int = 5) -> Optional[str]:
        """Get formatted context string for LLM prompt."""
        chunks = self.retrieve_chunks(query, k=k)
        if not chunks:
            return None
        return self._context_formatter(chunks)
