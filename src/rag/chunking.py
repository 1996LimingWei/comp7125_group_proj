import hashlib
import os
from typing import List, Tuple

from .types import Chunk


def load_documents(data_dir: str) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    try:
        filenames = sorted(os.listdir(data_dir))
    except Exception:
        return docs

    for filename in filenames:
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append((filename, f.read()))
        except Exception:
            continue

    return docs


def chunk_documents(
    docs: List[Tuple[str, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    step = max(1, int(chunk_size) - int(chunk_overlap))
    tokenizer = _get_gpt2_tokenizer()

    chunks: List[Chunk] = []
    for file_name, text in docs:
        chunk_index = 0
        if tokenizer is not None:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens).strip()
                if len(chunk_text) < 50:
                    continue

                cid = hashlib.sha256(
                    f"{file_name}:{i}:{i + len(chunk_tokens)}:{chunk_size}:{chunk_overlap}".encode("utf-8")
                ).hexdigest()
                chunks.append(Chunk(
                    id=cid,
                    file_name=file_name,
                    chunk_id=chunk_index,
                    start_token=i,
                    end_token=i + len(chunk_tokens),
                    text=chunk_text,
                ))
                chunk_index += 1
            continue

        words = text.split()
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words).strip()
            if len(chunk_text) < 50:
                continue

            cid = hashlib.sha256(
                f"{file_name}:{i}:{i + len(chunk_words)}:{chunk_size}:{chunk_overlap}".encode("utf-8")
            ).hexdigest()
            chunks.append(Chunk(
                id=cid,
                file_name=file_name,
                chunk_id=chunk_index,
                start_token=i,
                end_token=i + len(chunk_words),
                text=chunk_text,
            ))
            chunk_index += 1

    return chunks


def _get_gpt2_tokenizer():
    try:
        from transformers import GPT2TokenizerFast as GPT2Tokenizer
    except Exception:
        try:
            from transformers import GPT2Tokenizer
        except Exception:
            return None

    try:
        return GPT2Tokenizer.from_pretrained("gpt2")
    except Exception:
        return None
