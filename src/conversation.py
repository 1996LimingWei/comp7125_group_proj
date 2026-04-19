from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
import re
from collections import Counter


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class NormalizedSnippet:
    citation_key: str
    text: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 512

    def to_ollama_options(self) -> Dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "num_predict": int(self.max_tokens),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
        }


def resolve_generation_config(
    gen_params: Optional[Any] = None,
    *,
    defaults: Optional[GenerationConfig] = None,
) -> GenerationConfig:
    base = defaults or GenerationConfig()
    if gen_params is None:
        return base

    if isinstance(gen_params, GenerationConfig):
        return gen_params

    if not isinstance(gen_params, Mapping):
        raise TypeError("gen_params must be a dict-like object or GenerationConfig")

    temperature = gen_params.get("temperature", base.temperature)
    max_tokens = gen_params.get("max_tokens", base.max_tokens)

    if temperature is None:
        temperature = base.temperature
    if max_tokens is None:
        max_tokens = base.max_tokens

    return GenerationConfig(
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are BU Assistant, an intelligent campus assistant for HKBU (Hong Kong Baptist University). "
    "Answer user questions using only the provided context. "
    "If the context does not contain enough information, say you are not sure and that the knowledge base may not cover it."
)


def normalize_snippets(
    retrieval_output: Any,
    *,
    snippet_pool: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    def safe_key(raw: str) -> str:
        s = raw.strip()
        s = s.replace("[", "(").replace("]", ")")
        s = "_".join(s.split())
        return s

    def ensure_unique_keys(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[str, int] = {}
        for item in items:
            key = str(item.get("citation_key") or "")
            if key not in seen:
                seen[key] = 1
                continue
            seen[key] += 1
            item["citation_key"] = f"{key}-{seen[key]}"
        return items

    def build_key(meta: Mapping[str, Any], fallback_index: int) -> str:
        file_name = meta.get("file_name") or meta.get("source")
        chunk_id = meta.get("chunk_id")
        snippet_index = meta.get("snippet_index")

        if file_name is not None and chunk_id is not None:
            return safe_key(f"{file_name}#chunk{chunk_id}")
        if file_name is not None and snippet_index is not None:
            return safe_key(f"{file_name}#i{snippet_index}")
        if file_name is not None:
            return safe_key(str(file_name))
        return f"S{fallback_index + 1}"

    def coerce_text(obj: Any) -> Optional[str]:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Mapping):
            for k in ("text", "content", "document"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        return None

    normalized: List[Dict[str, Any]] = []

    if retrieval_output is None:
        return normalized

    if isinstance(retrieval_output, str):
        text = retrieval_output.strip()
        if not text:
            return normalized
        meta: Dict[str, Any] = {}
        normalized.append(
            NormalizedSnippet(citation_key="S1", text=text, meta=meta).__dict__
        )
        return normalized

    if isinstance(retrieval_output, Mapping) and "snippets" in retrieval_output:
        retrieval_output = retrieval_output.get("snippets")

    if not isinstance(retrieval_output, Sequence):
        text = coerce_text(retrieval_output)
        if not text:
            raise TypeError("retrieval_output must be a sequence of snippets/results")
        meta = {}
        normalized.append(
            NormalizedSnippet(citation_key="S1", text=text, meta=meta).__dict__
        )
        return normalized

    for i, item in enumerate(retrieval_output):
        meta: Dict[str, Any] = {}
        text: Optional[str] = None

        if isinstance(item, int):
            if snippet_pool is None:
                raise ValueError("snippet_pool is required when retrieval_output contains indices")
            if item < 0 or item >= len(snippet_pool):
                continue
            snippet = snippet_pool[item]
            text = coerce_text(snippet)
            meta.update(dict(snippet))
            meta["snippet_index"] = item
        elif isinstance(item, Mapping):
            text = coerce_text(item)
            meta.update(dict(item))
        elif isinstance(item, tuple) or isinstance(item, list):
            parts = list(item)
            idx = None
            if parts and isinstance(parts[-1], int):
                idx = parts[-1]

            if idx is not None and snippet_pool is not None:
                if 0 <= idx < len(snippet_pool):
                    snippet = snippet_pool[idx]
                    text = coerce_text(snippet)
                    meta.update(dict(snippet))
                    meta["snippet_index"] = idx
                if len(parts) >= 2:
                    meta["score"] = parts[0]
                if len(parts) >= 3:
                    meta["overlap_keywords"] = parts[1]
            else:
                text = coerce_text(parts[0]) if parts else None
                if len(parts) >= 2 and isinstance(parts[1], Mapping):
                    meta.update(dict(parts[1]))
        else:
            text = coerce_text(item)

        if not text:
            continue

        if "file_name" not in meta and "source" in meta:
            meta["file_name"] = meta.get("source")

        citation_key = build_key(meta, fallback_index=i)
        normalized.append(
            NormalizedSnippet(
                citation_key=citation_key,
                text=text.strip(),
                meta=meta,
            ).__dict__
        )

    return ensure_unique_keys(normalized)


def build_prompt(
    query: str,
    snippets: Any,
    history: Optional[List[Dict[str, str]]] = None,
    use_history: bool = False,
    max_history_messages: int = 12,
    system_instruction: Optional[str] = None,
) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if max_history_messages < 1:
        raise ValueError("max_history_messages must be >= 1")

    sys_inst = (system_instruction or DEFAULT_SYSTEM_INSTRUCTION).strip()

    normalized: List[Dict[str, Any]]
    if isinstance(snippets, Sequence) and not isinstance(snippets, str):
        items = list(snippets)
        if items and all(
            isinstance(x, Mapping) and "citation_key" in x and "text" in x for x in items
        ):
            normalized = [
                {
                    "citation_key": str(x.get("citation_key")),
                    "text": str(x.get("text") or "").strip(),
                    "meta": dict(x.get("meta") or {}),
                }
                for x in items
                if str(x.get("text") or "").strip()
            ]
        else:
            normalized = normalize_snippets(snippets)
    else:
        normalized = normalize_snippets(snippets)

    context_lines: List[str] = []
    for s in normalized:
        key = str(s.get("citation_key") or "").strip() or "S?"
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        context_lines.append(f"【{key}】 {text}")

    context_block = "\n\n".join(context_lines) if context_lines else "（未检索到相关上下文）"

    history_block = ""
    if use_history and history:
        sliced = history[-max_history_messages:]
        formatted: List[str] = []
        for m in sliced:
            if not isinstance(m, Mapping):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if not role or not content:
                continue
            if role == "user":
                role_label = "User"
            elif role == "assistant":
                role_label = "Assistant"
            else:
                continue
            formatted.append(f"{role_label}: {content}")
        if formatted:
            history_block = "\n\nConversation History:\n" + "\n".join(formatted)

    rules_block = (
        "Answer Rules:\n"
        "1) Use only the Context to answer.\n"
        "2) When you use a fact, cite it with the same citation key like 【...】.\n"
        "3) If the Context does not contain the answer, say you are not sure and that the knowledge base may not cover it.\n"
        "4) Do not invent details or sources.\n"
        "5) History is only for conversational continuity; factual claims must come from Context and be cited.\n"
    )

    prompt = (
        f"System Persona:\n{sys_inst}\n\n"
        f"Context:\n{context_block}"
        f"{history_block}\n\n"
        f"User Query:\n{query.strip()}\n\n"
        f"{rules_block}\n"
        "Based on the above, here is your answer:\n"
    )
    return prompt


@dataclass(frozen=True)
class SnippetRecord:
    citation_key: str
    file_name: Optional[str] = None
    chunk_id: Optional[int] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_key": self.citation_key,
            "file_name": self.file_name,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


@dataclass(frozen=True)
class GenerationRecord:
    session_id: Optional[str]
    query: str
    used_history_count: int
    snippets: List[SnippetRecord]
    prompt_chars: int
    answer_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "used_history_count": int(self.used_history_count),
            "snippets": [s.to_dict() for s in self.snippets],
            "prompt_chars": int(self.prompt_chars),
            "answer_text": self.answer_text,
        }


def build_generation_record(
    *,
    session_id: Optional[str],
    query: str,
    snippets: Any,
    prompt: str,
    answer_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    use_history: bool = False,
    max_history_messages: int = 12,
) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string")
    if max_history_messages < 1:
        raise ValueError("max_history_messages must be >= 1")

    normalized: List[Dict[str, Any]]
    if isinstance(snippets, Sequence) and not isinstance(snippets, str):
        items = list(snippets)
        if items and all(
            isinstance(x, Mapping) and "citation_key" in x and "text" in x for x in items
        ):
            normalized = [
                {
                    "citation_key": str(x.get("citation_key")),
                    "text": str(x.get("text") or "").strip(),
                    "meta": dict(x.get("meta") or {}),
                }
                for x in items
            ]
        else:
            normalized = normalize_snippets(snippets)
    else:
        normalized = normalize_snippets(snippets)

    snippet_records: List[SnippetRecord] = []
    for s in normalized:
        meta = s.get("meta") if isinstance(s, Mapping) else None
        meta_map = dict(meta) if isinstance(meta, Mapping) else {}

        citation_key = str(s.get("citation_key") or "").strip()
        file_name = meta_map.get("file_name") or meta_map.get("source")
        chunk_id = meta_map.get("chunk_id")
        score = meta_map.get("score")

        file_name_str = str(file_name) if file_name is not None else None

        chunk_id_int: Optional[int]
        if chunk_id is None:
            chunk_id_int = None
        else:
            try:
                chunk_id_int = int(chunk_id)
            except Exception:
                chunk_id_int = None

        score_float: Optional[float]
        if score is None:
            score_float = None
        else:
            try:
                score_float = float(score)
            except Exception:
                score_float = None

        if citation_key:
            snippet_records.append(
                SnippetRecord(
                    citation_key=citation_key,
                    file_name=file_name_str,
                    chunk_id=chunk_id_int,
                    score=score_float,
                )
            )

    used_history_count = 0
    if use_history and history:
        sliced = history[-max_history_messages:]
        for m in sliced:
            if not isinstance(m, Mapping):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if not role or not content:
                continue
            if role not in {"user", "assistant"}:
                continue
            used_history_count += 1

    record = GenerationRecord(
        session_id=session_id,
        query=query.strip(),
        used_history_count=used_history_count,
        snippets=snippet_records,
        prompt_chars=len(prompt),
        answer_text=answer_text,
    )
    return record.to_dict()


class ConversationManager:
    def __init__(
        self,
        *,
        system_message: Optional[str] = None,
        session_id: Optional[str] = None,
        max_turns: int = 6,
        summary_max_sentences: int = 3,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if summary_max_sentences < 1:
            raise ValueError("summary_max_sentences must be >= 1")

        self.session_id = session_id
        self.max_turns = max_turns
        self.system_message = system_message
        self.summary_max_sentences = summary_max_sentences
        self._messages: List[Message] = []
        self._summary: str = ""

    def add_user_message(self, text: str) -> None:
        self._append(role="user", content=text)

    def add_assistant_message(self, text: str) -> None:
        self._append(role="assistant", content=text)

    def get_history(self) -> List[Dict[str, str]]:
        history: List[Dict[str, str]] = []

        if self._summary.strip():
            history.append({
                "role": "assistant",
                "content": "Earlier conversation summary:\n" + self._summary.strip(),
            })

        history.extend([{"role": m.role, "content": m.content} for m in self._messages])
        return history

    def _append(self, *, role: str, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        self._messages.append(Message(role=role, content=content))
        self._truncate()

    def _truncate(self) -> None:
        max_non_system_messages = self.max_turns * 2
        if len(self._messages) <= max_non_system_messages:
            return

        overflow = self._messages[:-max_non_system_messages]
        self._messages = self._messages[-max_non_system_messages:]
        self._summary = _summarize_messages(
            previous_summary=self._summary,
            messages=overflow,
            max_sentences=self.summary_max_sentences,
        )


def _summarize_messages(
    *,
    previous_summary: str,
    messages: List[Message],
    max_sentences: int,
) -> str:
    chunks: List[str] = []
    if isinstance(previous_summary, str) and previous_summary.strip():
        chunks.append(previous_summary.strip())

    for m in messages:
        if m.role not in {"user", "assistant"}:
            continue
        content = m.content.strip()
        if not content:
            continue
        prefix = "User: " if m.role == "user" else "Assistant: "
        chunks.append(prefix + content)

    source = "\n".join(chunks).strip()
    if not source:
        return ""

    sentences = _split_sentences(source)
    if not sentences:
        return _hard_truncate(source, 480)

    token_freq = _token_frequencies(sentences)
    scored: List[tuple[float, int, str]] = []
    for idx, s in enumerate(sentences):
        tokens = _tokenize(s)
        score = float(sum(token_freq.get(t, 0) for t in tokens))
        if len(s) < 20:
            score *= 0.6
        scored.append((score, idx, s))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected_idx: set[int] = set()
    selected: List[tuple[int, str]] = []
    for _, idx, s in scored:
        if idx in selected_idx:
            continue
        selected_idx.add(idx)
        selected.append((idx, s))
        if len(selected) >= max_sentences:
            break

    selected.sort(key=lambda x: x[0])
    summary = " ".join(s.strip() for _, s in selected).strip()
    return _hard_truncate(summary, 600)


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = re.split(r"(?<=[\.\!\?\。\！\？])\s+", normalized)
    sentences: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if len(s) > 1000:
            s = s[:1000].strip()
        sentences.append(s)
    return sentences


def _tokenize(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", text.lower())
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "my",
        "your",
        "our",
        "their",
        "me",
        "us",
        "them",
    }
    return [t for t in raw if t and t not in stop]


def _token_frequencies(sentences: List[str]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for s in sentences:
        c.update(_tokenize(s))
    return dict(c)


def _hard_truncate(text: str, limit: int) -> str:
    s = text.strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "..."