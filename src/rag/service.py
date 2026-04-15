import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .chunking import chunk_documents, load_documents
from .embeddings import OllamaEmbedder, OllamaEmbeddingConfig, SentenceTransformerEmbedder
from .manifest import compute_manifest
from .types import Chunk, RetrievedChunk, retrieved_chunks_to_snippets
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Detect course codes like COMP7125 / DAAI1234. Used for query-time reranking.
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\d{4}\b")


def format_chunks_for_prompt(chunks: Sequence[RetrievedChunk]) -> str:
    # Convert retrieved chunks into a plain-text context block for prompt injection.
    parts = []
    for c in chunks:
        header = f"Source: {c.source}#chunk:{c.chunk_id} distance:{c.distance}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


class RAGService:
    # Orchestration layer: builds/refreshes the vector index and exposes retrieval helpers.
    def __init__(
        self,
        data_dir: str = "./course_docs",
        chroma_path: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        rebuild_if_changed: bool = True,
        ollama_base_url: str = "http://localhost:11434",
        ollama_embed_model: str = "nomic-embed-text",
        embedding_model: Optional[Any] = None,
        embedder: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        context_formatter: Optional[Callable[[Sequence[RetrievedChunk]], str]] = None,
    ):
        self.data_dir = data_dir
        self.chroma_path = chroma_path
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.rebuild_if_changed = bool(rebuild_if_changed)
        self._context_formatter = context_formatter or format_chunks_for_prompt

        # Embedding backend: default to Ollama embeddings, but allow dependency injection.
        if embedder is not None:
            self.embedder = embedder
        elif embedding_model is not None:
            self.embedder = SentenceTransformerEmbedder(
                embedding_model,
                embedding_id="sentence-transformers",
            )
        else:
            self.embedder = OllamaEmbedder(OllamaEmbeddingConfig(
                base_url=ollama_base_url,
                model=ollama_embed_model,
            ))

        # Vector store backend (Chroma by default); injectable for tests.
        self.vector_store = vector_store or ChromaVectorStore(chroma_path=chroma_path)

        if self.rebuild_if_changed:
            # Rebuild the index when source docs / params / embedding backend change.
            self._ensure_index()
        else:
            # Only build if empty; do not compute manifest / refresh.
            if self.vector_store.count() == 0:
                self._build_index()

    def count(self) -> int:
        return self.vector_store.count()

    def _rerank_course_query(
        self,
        *,
        query: str,
        chunks: List[RetrievedChunk],
        keep_k: int,
    ) -> List[RetrievedChunk]:
        # Lightweight hybrid rerank:
        # - Start with vector candidates
        # - Boost chunks that mention the detected course code(s) and prerequisite hints
        codes = set(_COURSE_CODE_RE.findall(query.upper()))
        if not codes:
            return chunks[:keep_k]

        query_l = query.lower()
        wants_prereq = ("prereq" in query_l) or ("prerequisite" in query_l)

        scored: List[tuple[float, int, RetrievedChunk]] = []
        for idx, c in enumerate(chunks):
            base = 0.0
            if isinstance(c.distance, (int, float)):
                base = 1.0 - float(c.distance)

            src_u = (c.source or "").upper()
            txt_u = (c.content or "").upper()

            boost = 0.0
            for code in codes:
                if code in src_u:
                    boost += 2.0
                if code in txt_u:
                    boost += 3.0

            if "COURSE_OUTLINE" in src_u:
                boost += 1.0

            if wants_prereq:
                txt_l = (c.content or "").lower()
                if ("prereq" in txt_l) or ("prerequisite" in txt_l):
                    boost += 1.5

            scored.append((base + boost, -idx, c))

        scored.sort(reverse=True)
        return [c for _, _, c in scored[:keep_k]]

    def retrieve_chunks(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        # Retrieve top-k chunks. For course-code queries we fetch a larger candidate set
        # and rerank using simple lexical cues (course code matches, "prerequisite", etc.).
        total = self.vector_store.count()
        if total == 0:
            return []
        query_emb = self.embedder.embed_query(query)
        k = int(k)
        has_course_code = bool(_COURSE_CODE_RE.search(query.upper()))
        if not has_course_code:
            return self.vector_store.query(query_embedding=query_emb, top_k=k)

        candidate_k = min(max(k * 5, 15), total)
        candidates = self.vector_store.query(query_embedding=query_emb, top_k=candidate_k)
        return self._rerank_course_query(query=query, chunks=candidates, keep_k=k)

    def retrieve_snippets(
        self,
        query: str,
        *,
        k: int = 3,
        citation_prefix: str = "N",
        retriever: str = "neural",
    ) -> List[Dict[str, Any]]:
        # Return Module 4/6 compatible snippets: [{"citation_key","text","meta"}...]
        chunks = self.retrieve_chunks(query, k=k)
        return retrieved_chunks_to_snippets(
            chunks,
            citation_prefix=citation_prefix,
            retriever=retriever,
        )

    def get_context(self, query: str, k: int = 5) -> Optional[str]:
        # Return a string context block (used by the CLI prompt-injection path).
        chunks = self.retrieve_chunks(query, k=k)
        if not chunks:
            return None
        return self._context_formatter(chunks)

    def _ensure_index(self) -> None:
        # Compute a manifest representing the docs+params+embedding backend.
        # If it differs from the stored manifest, rebuild the entire index.
        current = compute_manifest(
            data_dir=self.data_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_id=self.embedder.embedding_id,
        )
        stored = self.vector_store.get_manifest()
        if self.vector_store.count() == 0 or stored is None or stored != current:
            self.vector_store.reset()
            self._build_index(manifest_text=current)

    def _build_index(self, manifest_text: Optional[str] = None) -> None:
        # Full rebuild: load docs -> chunk -> embed -> write to vector store -> store manifest.
        docs = load_documents(self.data_dir)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return

        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        if manifest_text is None:
            manifest_text = compute_manifest(
                data_dir=self.data_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_id=self.embedder.embedding_id,
            )
        self.vector_store.upsert_manifest(
            manifest_text=manifest_text,
            embedding=self.embedder.embed_query(manifest_text),
        )
