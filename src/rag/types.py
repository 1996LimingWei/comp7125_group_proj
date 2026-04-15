from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Chunk:
    id: str
    file_name: str
    chunk_id: int
    start_token: int
    end_token: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    chunk_id: Optional[int]
    start_token: Optional[int]
    end_token: Optional[int]
    distance: Optional[float]


def retrieved_chunks_to_snippets(
    retrieved: Sequence["RetrievedChunk"],
    *,
    citation_prefix: str = "N",
    retriever: str = "neural",
) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    for i, chunk in enumerate(retrieved):
        citation_key = f"{citation_prefix}{i}"
        distance = chunk.distance
        score = None
        if isinstance(distance, (int, float)):
            score = 1.0 - float(distance)
        snippets.append({
            "citation_key": citation_key,
            "text": chunk.content,
            "meta": {
                "file_name": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": score,
                "distance": distance,
                "retriever": retriever,
            },
        })
    return snippets
