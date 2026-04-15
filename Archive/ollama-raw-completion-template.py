'''
Setup

Run Ollama before executing cells:
- `ollama serve`
- `ollama pull gemma3:4b` (or another <=12B local model)

Completion via Ollama SDK

This uses `ollama.generate(...)`, which wraps the same `/api/generate` endpoint.
It is still raw completion (not chat, i.e., `ollama.chat(...)`).

Make sure that `raw=True` inside the generate() function.

'''

MODEL = "gemma3:4b"  # change to any local completion-capable model

def complete_document_sdk(prefix: str, *, model: str = MODEL, num_predict: int = 180, temperature: float = 0.5) -> str:
    import ollama

    response = ollama.generate(
        model=model,
        prompt=prefix,
        stream=False,
        # raw shoudl be set to True to get the raw completion without any post-processing or prompt templating
        # more information about the raw option can be found in the API reference: https://docs.ollama.com/api/generate
        raw=True,
        options={
            "num_predict": num_predict,
            "temperature": temperature,
        },
    )
    return response["response"]

doc_prefix_sdk = """Question: What is the capital of France?

Answer: """

completion_sdk = complete_document_sdk(doc_prefix_sdk)
print(doc_prefix_sdk + completion_sdk)

completion_sdk

'''
Same Completion via Ollama API

Alternatively, this uses Ollama's `POST /api/generate` endpoint for plain document continuation.
It intentionally does **not** use `POST /api/chat`.
'''

from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"  # raw completion endpoint
MODEL = "gemma3:4b"  # change to any local completion-capable model

def complete_document(prefix: str, *, model: str = MODEL, num_predict: int = 180, temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "prompt": prefix,
        "stream": False,
        # raw shoudl be set to True to get the raw completion without any post-processing or prompt templating
        # more information about the raw option can be found in the API reference: https://docs.ollama.com/api/generate
        "raw": True,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

doc_prefix_api = """Question: What is the capital of the United States of America?

Answer: """

completion = complete_document(doc_prefix_api)
print(doc_prefix_api + completion)

'''
If you get `ConnectionError`, start Ollama first (`ollama serve`) and ensure your model is available (`ollama pull llama3.1:8b`).
If `requests` is missing, run `%pip install requests` in a notebook cell.
'''
