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
├── rag/service.py     # RAG service (ChromaDB)
├── ollama/chat.py     # LLM service (Ollama)
└── storage/mongo.py   # Storage service (Cosmos DB)
```
