# Project Updates by Liming 2026.03.30

## Key Changes

### 1. RAG (Retrieval-Augmented Generation) for Course Data
**What:** Implemented semantic search over course documentation using ChromaDB vector database.

**How it works:**
- Integrate the Pan Junle's course documents in `course_docs/` are automatically chunked into 512-token snippets
- Each chunk is converted to vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- On user query, retrieves top-5 most relevant chunks
- Injects retrieved context into LLM prompt for grounded responses

**Files:** `src/rag/service.py`

**Test:** Run `python main.py` to see RAG retrieval in action

---

### 2. Set up Ollama Local LLM Integration
**What:** Initialize the AI assistant with local Ollama model (`gemma3:4b`).

**Configuration:**
```bash
# Ensure Ollama is running
ollama serve
ollama pull gemma3:4b
```

**Files:** `src/ollama/chat.py`, `main.py`

---

### 3. Cosmos DB Conversation History
**What:** Storage of all conversations using Azure Cosmos DB (MongoDB API).

**Features:**
- Stores messages with session_id, timestamp, user_id, role, content
- Automatically summarizes long conversations (>6 messages) to avoid token limits
- Graceful degradation: works without database if connection fails

**Connection String:** Set `MONGODB_URI` in `.env` file

**Database/Collection:** `7125Bot` / `ChatMessages`

**Files:** `src/storage/mongo.py`

---



### 4. Updated Modular Architecture
**Project Structure:**
```
src/
├── config.py          # Configuration management
├── cli/main.py        # CLI entry point
├── conversation.py    # Prompt building + conversation utilities (Module 4)
├── rag/service.py     # RAG service (ChromaDB)
├── ollama/chat.py     # LLM service (Ollama)
└── storage/mongo.py   # Storage service (Cosmos DB)
```

---

### 5. Module 4: Conversation Manager & Prompt Engineering (In Progress)
**What:** Added a minimal, self-contained Module 4 foundation for prompt assembly, conversation history handling, and structured generation records.

**Includes:**
- In-memory `ConversationManager` with turn-based truncation (keeps only the latest N turns).
- `normalize_snippets(...)` adapter to standardize retrieval outputs into a stable snippet format with citation keys.
- `build_prompt(...)` prompt assembler with fixed structure and citation enforcement; history injection is optional and length-limited.
- `GenerationConfig` and `resolve_generation_config(...)` for lightweight generation parameter packing/override.
- `build_generation_record(...)` to return a structured record for later evaluation modules (no persistence inside Module 4).

**Files:** `src/conversation.py`

---

## Updates (2026.04.05)

### Module 4: Foundation Completed
**What changed:**
- Added a standalone prompt/conversation utility module to support consistent prompt assembly and later evaluation logging, without changing existing RAG/Ollama/storage flows.

**Public APIs (src/conversation.py):**
- `ConversationManager(system_message=None, session_id=None, max_turns=6)`
  - `add_user_message(text)`, `add_assistant_message(text)`, `get_history() -> [{role, content}]`
- `normalize_snippets(retrieval_output, snippet_pool=None) -> [{citation_key, text, meta}]`
  - Accepts flexible retriever outputs (dict/tuple/index/string) and produces stable citation keys.
- `build_prompt(query, snippets, history=None, use_history=False, max_history_messages=12, system_instruction=None) -> str`
  - Fixed prompt structure and citation enforcement; history injection is optional and bounded.
- `GenerationConfig(temperature=0.7, max_tokens=512)` and `resolve_generation_config(gen_params=None, defaults=None)`
  - Packs and overrides generation params; can map to Ollama options via `to_ollama_options()`.
- `build_generation_record(session_id, query, snippets, prompt, answer_text, history=None, use_history=False, max_history_messages=12) -> dict`
  - Returns a structured record: `session_id`, `query`, `used_history_count`, `snippets`, `prompt_chars`, `answer_text`.
