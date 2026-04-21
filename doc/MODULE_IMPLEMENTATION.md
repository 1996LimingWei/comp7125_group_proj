# Module Implementation Documentation

This document maps the project's minimum and advanced implementation requirements to their corresponding code locations.

---

## Minimum Implementations

### 1. Working Baseline in Jupyter Notebook (Lab Sessions)
**Requirement:** A working baseline in at least one Jupyter Notebook that runs end-to-end.

**Implementation:**
| Notebook | Purpose | Location |
|----------|---------|----------|
| Module 1 Data Ingestion & Chunking | Document loading, tokenization, chunking | `Archive/Module1 Data Ingestion & Chunking.ipynb` |
| Module 2 Lexical Retrieval | BM25 search and RAG pipeline | `Archive/Module2_LexicalRetrieval.ipynb` |
| Ollama Raw Completion | LLM generation template | `Archive/ollama-raw-completion-template.ipynb` |

---

### 2. Prompt Assembly (Topic 6)
**Requirement:** Prompt assembly with clear instruction, context, and response-format control (role/task framing, explicit constraints, and a clear transition to final answer).

**Implementation:**
- **File:** `src/conversation.py` (lines 206-290)
- **Function:** `build_prompt()`
- **Structure:**
  - System Persona (role framing)
  - Context block with citations【filename#chunk:id】
  - Conversation History (if enabled)
  - User Query
  - Answer Rules (explicit constraints: use only context, cite sources, no hallucination)
  - Transition phrase: "Based on the above, here is your answer:"

---

### 3. Generation Control (Topic 2)
**Requirement:** Generation control with key parameters (temperature and output length) and short justification of settings in the report.

**Implementation:**
- **File:** `src/conversation.py` (lines 23-66)
- **Class:** `GenerationConfig`
- **Parameters:**
  - `temperature: float = 0.7` (0.3 for factual RAG answers, 0.7 for conversational creativity)
  - `max_tokens: int = 512` (180 for short answers, 512 for detailed responses, 1024 for study plans)
- **Justification:** Lower temperature ensures factual consistency in RAG; higher values allow conversational flow. Token limits prevent excessive generation and control response length per use case.

---

### 4. Conversation Management (Topic 3)
**Requirement:** Conversation management where applicable (history handling and/or system-message control).

**Implementation:**
- **File:** `src/conversation.py` (lines 433-491)
- **Class:** `ConversationManager`
- **Features:**
  - Message history with `add_user_message()` / `add_assistant_message()`
  - Automatic truncation after `max_turns=6` (12 messages)
  - Conversation summarization for overflow messages
  - System message injection via `system_prompt` parameter

---

### 5. Context Engineering Pipeline (Topic 4&5)
**Requirement:** Context engineering pipeline: retrieval, snippetizing/chunking, and relevance prioritization before generation.

**Implementation:**
| Stage | File | Function/Class | Description |
|-------|------|----------------|-------------|
| Chunking | `Module1 Data Ingestion & Chunking.py` | `chunk_documents()` | 512-token chunks with 50-token overlap |
| Retrieval (Lexical) | `Module2_LexicalRetrieval.py` | `lexical_search()` | BM25Okapi keyword matching |
| Retrieval (Neural) | `src/rag/service.py` | `retrieve_chunks()` | ChromaDB vector similarity with nomic-embed-text |
| Relevance Prioritization | `src/rag/service.py` (lines 77-121) | `_rerank_course_query()` | Course-code detection + lexical boosting (+3.0 for code in text, +1.5 for prerequisites) |
| Context Formatting | `src/rag/service.py` (lines 17-23) | `format_chunks_for_prompt()` | Converts chunks to plain-text context block |

---

### 6. RAG Evidence: Lexical vs Neural Retrieval (Topic 5)
**Requirement:** RAG evidence with at least one lexical retrieval setting and one embedding-based retrieval setting, plus a short comparison.

**Implementation:**
| Setting | File | Method | Model/Algorithm |
|---------|------|--------|----------------|
| Lexical | `Module2_LexicalRetrieval.py` | `lexical_search()` | BM25Okapi (rank-bm25 library) |
| Neural | `src/rag/service.py` | `retrieve_chunks()` | nomic-embed-text via Ollama + ChromaDB |

**Comparison:**
- **Lexical (BM25):** Excels at exact keyword matching, course code detection, fast inference
- **Neural (Embeddings):** Excels at semantic understanding, conceptual similarity, handles paraphrasing

---

### 7. Basic Evaluation (Topic 4)
**Requirement:** Basic evaluation on representative queries, including at least one quality comparison and one token-usage comparison.

**Implementation:**
- **Directory:** `module6_evaluation/`
- **Notebook:** `Module 6 Evaluation & Analysis.ipynb`
- **Utilities:** `module6_evaluation/evaluation_utils.py`
- **Framework:**
  - Three approaches compared: No-RAG baseline, Lexical (BM25), Neural (embeddings)
  - LLM-as-a-judge with metrics: relevance, correctness, helpfulness (1-5 scale)
  - Token usage tracking via `prompt_chars` in `GenerationRecord`
- **Results:** `module6_evaluation/results/` (CSV files + comparison plots)

---

## Advanced Implementations

### 1. Conversational Workflows / Agents
**Requirement:** Any technique from Topics 7-11, including conversational workflows/agents.

**Implementation:**
- **File:** `src/study_plan/manager.py`
- **Class:** `StudyPlanManager`
- **Features:**
  - Multi-turn stateful dialogue with state machine (`collecting_constraints` → `ready_to_generate` → `generating`)
  - Constraint collection across 4 dimensions: hours, goals, workload, days
  - Intent detection via keyword matching (`is_study_plan_query()`)
  - Course-specific retrieval integration for context-aware planning

---

### 2. Web / Mobile UI
**Requirement:** Any technique outside the taught syllabus (web/mobile UI).

**Implementation:**
- **Directory:** `web_ui/`
- **Stack:** React 18 + modern CSS
- **Features:**
  - ChatGPT-like interface with sidebar navigation
  - Markdown rendering with code block support
  - Generation control sliders (temperature, max_tokens)
  - Quick actions (Study Plan, Programs)
  - Responsive design for desktop and mobile
- **Backend:** Flask API (`web_app.py`) with proxy to `HKBUAssistant` class

---

### 3. Database-Backed Product Features
**Requirement:** Any technique outside the taught syllabus (database-backed product features).

**Implementation:**
- **File:** `src/storage/mongo.py`
- **Class:** `CosmosDBStorage`
- **Features:**
  - Session persistence with unique session IDs
  - Conversation history storage and retrieval
  - Resume interrupted conversations
  - Graceful degradation when DB is unavailable

---

### 4. Deployment-Ready Architecture
**Requirement:** Any technique outside the taught syllabus (deployment, external orchestration).

**Implementation:**
- **File:** `src/config.py`
- **Features:**
  - Environment-based configuration (`.env` support)
  - Modular service architecture with dependency injection
  - Dual-interface deployment: CLI (`src/cli/main.py`) + Web (`web_app.py`)
  - Graceful degradation for all external services (Ollama, CosmosDB)

---

## Module Details

### Data Ingestion & Chunking
**Assigned to:** PAN Junle  
**File:** `Module1 Data Ingestion & Chunking.py`

**Key Functions:**
- `load_documents(folder_path)` - Scans directory for `.txt` files
- `chunk_documents(docs, chunk_size=512, overlap=50)` - GPT2Tokenizer-based sliding window chunking
- `save_snippets()` / `load_snippets()` - JSON persistence

**Output:** `./output/snippets.json` (14 documents → ~32 chunks)

---

### Lexical Retrieval
**Assigned to:** Ke Linyao  
**File:** `Module2_LexicalRetrieval.py`

**Key Functions:**
- `lexical_search(query, snippets, top_k=3)` - BM25Okapi ranking
- `answer_with_lexical_rag(query, snippets, top_k=3)` - End-to-end RAG pipeline
- `ollama_generate(prompt)` - LLM generation with `raw=True`

**Dependencies:** `rank_bm25`, `ollama`

---

### Neural Retrieval
**Assigned to:** CHENG Dianni  
**Files:** `src/rag/service.py`, `src/rag/embeddings.py`, `src/rag/vector_store.py`

**Key Components:**
- `RAGService` - Orchestration layer with reranking and context formatting
- `OllamaEmbedder` - nomic-embed-text embedding generation
- `ChromaVectorStore` - Persistent vector storage with manifest-based rebuild

---

### Conversation Manager & Prompt Engineering
**Assigned to:** Cheng Hongliang  
**File:** `src/conversation.py`

**Key Components:**
- `ConversationManager` - History management with truncation and summarization
- `build_prompt()` - Structured prompt assembly with citations and rules
- `GenerationConfig` - Configurable temperature and max_tokens

---

### Study Plan Feature
**Assigned to:** Liming Wei  
**File:** `src/study_plan/manager.py`

**Key Components:**
- `StudyPlanManager` - State machine for constraint collection
- `UserConstraints` - Dataclass for structured constraint storage
- `_parse_constraints()` - Natural language parsing for hours, workload, days, courses, goals
- `generate_study_plan()` - Retrieval-augmented plan generation with specialized prompting

---

## Module Integration

### How Modules Connect

```
Data Ingestion → Retrieval (Lexical/Neural) → Prompt → Ollama
     ↓                    ↓                      ↓
  snippets.json    lexical_search()      build_prompt()
  chunks in DB     neural_search()       ConversationManager
```

### Integration in `main.py`

```python
# Load data
snippets = load_snippets("./output/snippets.json")

# Initialize RAG (neural retrieval)
rag_service = RAGService(data_dir="./course_docs")

# Conversation manager
conv_manager = ConversationManager()

# Query flow:
# 1. Retrieve context
context = rag_service.get_context(query)

# 2. Build prompt
prompt = build_prompt(query, context, conv_manager.get_history())

# 3. Generate with Ollama
response = ollama_client.chat(prompt)

# 4. Update conversation
conv_manager.add_user_message(query)
conv_manager.add_assistant_message(response)
```

---

## Testing Each Module

### Data Ingestion & Chunking
```bash
python "Module1 Data Ingestion & Chunking.py"
# Output: ./output/snippets.json
```

### Lexical Retrieval
```bash
python Module2_LexicalRetrieval.py
# Tests lexical_search() and answer_with_lexical_rag()
```

### Neural Retrieval
```bash
python main.py
# Uses RAGService for neural retrieval
```

### Conversation Manager
```bash
python -m src.cli.main
# Interactive CLI with conversation history
```

### Study Plan Feature
```bash
python test_study_plan.py
# Tests StudyPlanManager constraint collection and plan generation
```

Or in CLI:
```
You: study plan
Assistant: I'll help you create a personalized study plan!...
```

---

## File Structure

```
comp7125_group_proj/
├── Module1 Data Ingestion & Chunking.py  # Data Ingestion
├── Module2_LexicalRetrieval.py           # Lexical Retrieval
├── src/
│   ├── rag/
│   │   ├── service.py                    # Neural Retrieval / RAG Orchestration
│   │   ├── embeddings.py                 # OllamaEmbedder (nomic-embed-text)
│   │   ├── vector_store.py               # ChromaVectorStore
│   │   └── neural_search.py              # Standalone neural retrieval
│   ├── conversation.py                   # Conversation Manager / Prompt Assembly
│   ├── study_plan/
│   │   └── manager.py                    # Study Plan Feature
│   ├── storage/
│   │   └── mongo.py                      # Cosmos DB Persistence
│   ├── cli/
│   │   └── main.py                       # CLI Interface / HKBUAssistant
│   └── config.py                         # Configuration Management
├── web_ui/                               # React Web Interface
├── web_app.py                            # Flask Backend API
├── test_study_plan.py                    # Study Plan test
├── main.py                               # Integration test
└── doc/
    ├── SETUP_GUIDE.md
    └── MODULE_IMPLEMENTATION.md          # This file
```

---

## Summary

| Component | Assigned To | Status | Key Functions |
|-----------|-------------|--------|---------------|
| Data Ingestion | PAN Junle | ✅ Complete | `load_documents`, `chunk_documents`, `load_snippets` |
| Lexical Retrieval | Ke Linyao | ✅ Complete | `lexical_search`, `answer_with_lexical_rag` |
| Neural Retrieval | CHENG Dianni | ✅ Complete | `RAGService.retrieve_chunks`, `get_context` |
| Conversation Manager | Cheng Hongliang | ✅ Complete | `ConversationManager`, `build_prompt`, `GenerationConfig` |
| Study Plan Feature | Liming Wei | ✅ Complete | `StudyPlanManager`, `generate_study_plan`, constraint parsing |
| Web UI | Liming Wei | ✅ Complete | React frontend, Flask backend |
| Database | Liming Wei | ✅ Complete | `CosmosDBStorage`, session persistence |
| Evaluation | Team | ✅ Complete | LLM-as-a-judge, quality/token comparison |

All minimum and advanced implementations are complete and integrated into the HKBU Course Assistant system.
