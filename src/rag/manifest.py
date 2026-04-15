import hashlib
import json
import os
from typing import Optional, Sequence, Dict, Any


def compute_manifest(
    *,
    data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_id: str,
) -> str:
    items = []
    try:
        filenames = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".txt")
        )
    except Exception:
        filenames = []

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            digest = hashlib.sha256(content).hexdigest()
            items.append({"file_name": filename, "sha256": digest})
        except Exception:
            items.append({"file_name": filename, "sha256": None})

    manifest = {
        "data_dir": os.path.abspath(data_dir),
        "files": items,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "embedding_id": str(embedding_id),
    }
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True)


def extract_manifest_document(documents: Any) -> Optional[str]:
    if not documents:
        return None
    if isinstance(documents, list) and documents:
        first = documents[0]
        if isinstance(first, str) and first:
            return first
    return None
