from __future__ import annotations

import re
from typing import List, Optional

import requests

from .chunking import load_documents, chunk_documents
from .embeddings import OllamaEmbedder, OllamaEmbeddingConfig
from .manifest import compute_manifest
from .types import RetrievedChunk
from .vector_store import ChromaVectorStore

# Detect course codes like COMP7125 / DAAI1234. Used for query-time reranking.
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\d{4}\b")

MODEL = "gemma3:4b"


def ollama_generate(
    prompt: str,
    *,
    model: str = MODEL,
    base_url: str = "http://localhost:11434",
    num_predict: int = 180,
    temperature: float = 0.3,
    timeout_s: int = 120,
) -> str:
    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": int(num_predict),
                "temperature": float(temperature),
            },
        },
        timeout=float(timeout_s),
    )
    r.raise_for_status()
    return str(r.json().get("response") or "").strip()


def answer_with_neural_rag(
    query: str,
    *,
    top_k: int = 3,
    data_dir: str = "./course_docs",
    chroma_path: str = "./chroma_db",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    rebuild_if_changed: bool = True,
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "nomic-embed-text",
    model: str = MODEL,
) -> str:
    retrieved = neural_search(
        query,
        top_k=top_k,
        data_dir=data_dir,
        chroma_path=chroma_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rebuild_if_changed=rebuild_if_changed,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
    )
    if not retrieved:
        return "No relevant information found."

    context_parts: List[str] = []
    for i, c in enumerate(retrieved):
        source = str(c.source or "Unknown")
        chunk_id = c.chunk_id if c.chunk_id is not None else "?"
        header = f"Source: {source}#chunk:{chunk_id}"
        context_parts.append(f"[Snippet {i+1}] {header}\n{c.content}")
    context = "\n\n".join(context_parts)

    prompt = f"""Use only the context below to answer the question.
Cite snippet numbers like [1][2].

Context:
{context}

Question: {query}
Answer:"""

    return ollama_generate(prompt, model=model, base_url=ollama_base_url)


def neural_search(
    query: str,
    *,
    top_k: int = 3,
    data_dir: str = "./course_docs",
    chroma_path: str = "./chroma_db",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    rebuild_if_changed: bool = True,
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "nomic-embed-text",
) -> List[RetrievedChunk]:
    # Standalone neural retrieval entry point (no RAGService instance required).
    # Builds/refreshes the index (optional) and returns RetrievedChunk objects.
    embedder = OllamaEmbedder(OllamaEmbeddingConfig(
        base_url=ollama_base_url,
        model=ollama_embed_model,
    ))
    store = ChromaVectorStore(chroma_path=chroma_path)
    _ensure_index(
        store=store,
        embedder=embedder,
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rebuild_if_changed=rebuild_if_changed,
    )
    query_emb = embedder.embed_query(query)
    top_k = int(top_k)
    has_course_code = bool(_COURSE_CODE_RE.search(query.upper()))
    if not has_course_code:
        return store.query(query_embedding=query_emb, top_k=top_k)

    # For course-code queries, rerank vector candidates using lexical cues.
    total = store.count()
    candidate_k = min(max(top_k * 5, 15), total)
    candidates = store.query(query_embedding=query_emb, top_k=candidate_k)
    codes = set(_COURSE_CODE_RE.findall(query.upper()))
    query_l = query.lower()
    wants_prereq = ("prereq" in query_l) or ("prerequisite" in query_l)

    scored: List[tuple[float, int, RetrievedChunk]] = []
    for idx, c in enumerate(candidates):
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
    return [c for _, _, c in scored[:top_k]]


def _ensure_index(
    *,
    store: ChromaVectorStore,
    embedder: OllamaEmbedder,
    data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    rebuild_if_changed: bool,
) -> None:
    # Ensure the Chroma index exists and is up-to-date (manifest-based rebuild).
    if not rebuild_if_changed:
        if store.count() == 0:
            _build_index(
                store=store,
                embedder=embedder,
                data_dir=data_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return

    current = compute_manifest(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_id=embedder.embedding_id,
    )
    stored = store.get_manifest()
    if store.count() == 0 or stored is None or stored != current:
        store.reset()
        _build_index(
            store=store,
            embedder=embedder,
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            manifest_text=current,
        )


def _build_index(
    *,
    store: ChromaVectorStore,
    embedder: OllamaEmbedder,
    data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    manifest_text: Optional[str] = None,
) -> None:
    # Full rebuild: load docs -> chunk -> embed -> write to vector store -> store manifest.
    docs = load_documents(data_dir)
    chunks = chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        return

    embeddings = embedder.embed_texts([c.text for c in chunks])
    store.add_chunks(chunks=chunks, embeddings=embeddings)

    if manifest_text is None:
        manifest_text = compute_manifest(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_id=embedder.embedding_id,
        )

    store.upsert_manifest(
        manifest_text=manifest_text,
        embedding=embedder.embed_query(manifest_text),
    )
