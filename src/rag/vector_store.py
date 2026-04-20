from __future__ import annotations

from typing import List, Optional

from .types import Chunk, RetrievedChunk
from .manifest import extract_manifest_document


COLLECTION_NAME = "hkbu_knowledge"
MANIFEST_ID = "__manifest__"


class ChromaVectorStore:
    # Thin wrapper around ChromaDB collection operations used by the RAG pipeline.
    def __init__(self, *, chroma_path: str):
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as e:
            raise ImportError("chromadb is required to use ChromaVectorStore") from e

        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get_manifest(self) -> Optional[str]:
        # Manifest is stored as a special document to detect index staleness.
        try:
            data = self._collection.get(ids=[MANIFEST_ID], include=["documents"])
            return extract_manifest_document(data.get("documents"))
        except Exception:
            return None

    def upsert_manifest(self, *, manifest_text: str, embedding: List[float]) -> None:
        # Store manifest as a non-chunk doc so it can be retrieved by ID and excluded from queries.
        self._collection.add(
            documents=[manifest_text],
            ids=[MANIFEST_ID],
            metadatas=[{"doc_type": "manifest"}],
            embeddings=[embedding],
        )

    def add_chunks(self, *, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        # Add chunk documents with metadata needed for traceability/citations.
        self._collection.add(
            documents=[c.text for c in chunks],
            ids=[c.id for c in chunks],
            metadatas=[{
                "doc_type": "chunk",
                "file_name": c.file_name,
                "chunk_id": c.chunk_id,
                "start_token": c.start_token,
                "end_token": c.end_token,
            } for c in chunks],
            embeddings=embeddings,
        )

    def query(self, *, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        # Query chunks only. We also defensively filter non-chunk docs.
        if self.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(int(top_k), self.count()),
            include=["documents", "metadatas", "distances"],
            where={"doc_type": "chunk"},
        )

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        retrieved: List[RetrievedChunk] = []
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
