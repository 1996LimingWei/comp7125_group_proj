# Module Implementation Documentation

This document describes how each module in the HKBU Course Assistant project was implemented and how they work together.

---

## Data Ingestion & Chunking
**Assigned to:** PAN Junle  
**File:** `Module1 Data Ingestion & Chunking.py`

### Responsibilities
- Collect local course documents and convert to plain text
- Chunk documents into 512-token snippets with 50-token overlap
- Store snippets in JSON format with metadata
- Provide `load_snippets()` function for retrieval modules

### Implementation Details

#### `load_documents(folder_path)`
```python
def load_documents(folder_path):
    """Reads all .txt files and returns list of {file_name, text} dicts."""
```
- Scans directory for `.txt` files
- Reads each file with UTF-8 encoding
- Returns list of documents with filename and content

#### `chunk_documents(docs, chunk_size=512, overlap=50)`
```python
def chunk_documents(docs, chunk_size=512, overlap=50):
    """Splits documents into token-based chunks with overlap."""
```
- Uses GPT2Tokenizer for consistent tokenization
- Implements sliding window with configurable overlap
- Assigns unique `chunk_id` to each snippet
- Returns list of snippets with metadata

#### `save_snippets(snippets, save_path)` & `load_snippets(load_path)`
- Saves/loads snippets to/from JSON
- JSON format: `[{"file_name": "...", "chunk_id": 0, "text": "..."}, ...]`
- Default path: `./output/snippets.json`

### Output
- Creates `./output/snippets.json` with all chunked documents
- Total: 14 documents → 32 chunks

---

## Lexical Retrieval
**Assigned to:** Ke Linyao  
**File:** `Module2_LexicalRetrieval.py`

### Responsibilities
- Implement `lexical_search(query, snippets, top_k=3)`
- Return list of `(score, overlap_keywords, snippet_index)` tuples
- Use BM25 for keyword matching
- Provide `answer_with_lexical_rag(query)` for end-to-end RAG

### Implementation Details

#### `load_snippets(load_path)`
- Loads chunked snippets from data ingestion output
- Default path: `./output/snippets.json`

#### `preprocess_text(text)`
- Simple tokenization: lowercase, remove punctuation, split by whitespace
- Used for both query and document processing

#### `lexical_search(query, snippets, top_k=3)`
```python
def lexical_search(query: str, snippets: List[dict], top_k: int = 3) -> List[Tuple[float, List[str], int]]:
    """Returns list of (score, overlap_keywords, snippet_index)."""
```
**Algorithm:**
1. Tokenize query and all snippet texts
2. Build BM25 index using `rank_bm25.BM25Okapi`
3. Score all snippets against query
4. Compute keyword overlap (intersection of query/doc tokens)
5. Sort by score descending, return top-k

**Return format:**
```python
[
    (1.2345, ["academic", "advising"], 5),
    (0.9876, ["advising", "student"], 12),
    (0.7654, ["academic"], 8),
]
```

#### `ollama_generate(prompt)`
- Calls Ollama with `raw=True` for completion-style generation
- Model: `gemma3:4b`
- Options: `num_predict=180`, `temperature=0.3`

#### `answer_with_lexical_rag(query, snippets, top_k=3)`
```python
def answer_with_lexical_rag(query: str, snippets: List[dict], top_k: int = 3) -> str:
    """Full lexical RAG pipeline: retrieve → prompt → generate."""
```
**Pipeline:**
1. Call `lexical_search()` to get top-k snippets
2. Build context with snippet citations `[Snippet 1]`, `[Snippet 2]`, etc.
3. Construct prompt with context and query
4. Call Ollama API with constructed prompt
5. Return generated answer

### Usage
```python
from Module2_LexicalRetrieval import lexical_search, answer_with_lexical_rag, load_snippets

# Load snippets
snippets = load_snippets("./output/snippets.json")

# Search
results = lexical_search("What programs does HKBU offer?", snippets, top_k=3)

# Full RAG
answer = answer_with_lexical_rag("What programs does HKBU offer?", snippets)
print(answer)
```

### Dependencies
```
rank_bm25
ollama
```

---

## Neural Retrieval
**Assigned to:** CHENG Dianni  
**File:** `src/rag/service.py`

### Responsibilities
- Use embeddings to create vector representations of snippets
- Build vector index using ChromaDB
- Implement `neural_search(query, top_k=3)`
- Provide `answer_with_neural_rag(query)`

### Implementation Details

#### `RAGService` Class
```python
class RAGService:
    def __init__(self, data_dir, chroma_path, chunk_size=512, ...):
        # Initialize embedding model (sentence-transformers)
        # Connect to ChromaDB
        # Build/load knowledge base
```

**Embedding Model:**
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Cosine similarity for retrieval

**Vector Index (ChromaDB):**
- Persistent storage in `./chroma_db/`
- Collection name: `"hkbu_knowledge"`
- Metadata: file_name, chunk_id, start_token, end_token
- Automatic rebuild if source documents change (SHA256 manifest)

#### `retrieve_chunks(query, k=5)`
```python
def retrieve_chunks(self, query, k=5):
    """Returns list of RetrievedChunk objects."""
```
**Algorithm:**
1. Encode query to embedding vector
2. Query ChromaDB with `query_embeddings`
3. Filter by `doc_type="chunk"`
4. Return chunks with metadata and distance scores

#### `get_context(query, k=5)`
- Retrieves chunks and formats them for prompt
- Format: `Source: filename#chunk:id distance:score\ncontent`

### Usage
```python
from src.rag.service import RAGService

# Initialize (builds index if needed)
rag = RAGService(data_dir="./course_docs", chroma_path="./chroma_db")

# Search
chunks = rag.retrieve_chunks("What programs does HKBU offer?", k=3)

# Get formatted context
context = rag.get_context("What programs does HKBU offer?", k=3)
```

---

## Conversation Manager & Prompt Engineering
**Assigned to:** Cheng Hongliang  
**File:** `src/conversation.py`

### Responsibilities
- `ConversationManager` class for message history
- Methods: `add_user_message`, `add_assistant_message`, `get_history`
- Prompt assembly with context, citations, and history
- Generation parameters (temperature, max_tokens)

### Implementation Details

#### `ConversationManager` Class
```python
class ConversationManager:
    def __init__(self, system_message=None, session_id=None, max_turns=6):
        # Initialize with optional system message
        # max_turns limits conversation history (6 turns = 12 messages)
```

**Methods:**
- `add_user_message(text)` - Add user message to history
- `add_assistant_message(text)` - Add assistant response to history
- `get_history()` - Returns list of `{"role": "...", "content": "..."}` dicts
- `_truncate()` - Automatically truncates history to max_turns

#### `build_prompt(query, snippets, history, ...)`
```python
def build_prompt(query, snippets, history=None, use_history=False, ...):
    """Constructs final prompt with system instruction, context, history, query."""
```

**Prompt Structure:**
```
System Persona:
{system_instruction}

Context:
【filename#chunk:id】 snippet text
【filename#chunk:id】 snippet text
...

Conversation History: (if enabled)
User: ...
Assistant: ...

User Query:
{query}

Answer Rules:
1) Use only the Context to answer.
2) When you use a fact, cite it with the same citation key like 【...】.
3) If the Context does not contain the answer, say you are not sure...
4) Do not invent details or sources.
5) History is only for conversational continuity...

Based on the above, here is your answer:
```

#### `GenerationConfig` Class
```python
@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 512
```

- `to_ollama_options()` - Converts to Ollama API format
- `resolve_generation_config()` - Merges user params with defaults

#### `normalize_snippets(retrieval_output, snippet_pool=None)`
- Normalizes various snippet formats to standard structure
- Handles: indices, dicts, tuples, strings
- Generates unique citation keys like `filename#chunk0`

### Usage
```python
from src.conversation import ConversationManager, build_prompt, GenerationConfig

# Create manager
conv = ConversationManager(system_message="You are a helpful assistant.")

# Add messages
conv.add_user_message("What programs does HKBU offer?")
conv.add_assistant_message("HKBU offers...")

# Build prompt
prompt = build_prompt(
    query="Tell me more about MSc programs",
    snippets=retrieved_snippets,
    history=conv.get_history(),
    use_history=True,
)

# Generation config
config = GenerationConfig(temperature=0.7, max_tokens=512)
```

---

## Study Plan Feature
**Assigned to:** Liming Wei  
**File:** `src/study_plan/manager.py`

### Responsibilities
- Design dialogue flow to collect user constraints (time, goals, workload)
- Use retrieval to fetch relevant course information
- Build specialized prompt for study plan generation
- Integrate with conversation manager for study-plan queries

### Implementation Details

#### `StudyPlanManager` Class
```python
class StudyPlanManager:
    def __init__(self, retrieval_service=None):
        # Initialize with lexical or neural retrieval service
        # States: collecting_constraints, ready_to_generate, complete
```

**Key Methods:**
- `start_study_plan_flow()` - Initiates constraint collection dialogue
- `collect_constraint(user_input)` - Parses and stores user constraints
- `generate_study_plan(snippets, ollama_client)` - Generates personalized plan
- `is_study_plan_query(text)` - Detects study plan requests

#### `UserConstraints` Dataclass
```python
@dataclass
class UserConstraints:
    available_hours_per_week: Optional[int]
    goals: List[str]
    workload_preference: WorkloadPreference  # light/moderate/heavy
    preferred_days: List[str]
    course_codes: List[str]
```

#### Constraint Parsing (`_parse_constraints`)
Parses natural language to extract:
- **Hours**: "15 hours per week" → `15`
- **Workload**: "light", "moderate", "heavy"
- **Days**: "Monday to Friday" → `["Monday", "Tuesday", ...]`
- **Courses**: "COMP7125" → `["COMP7125"]`
- **Goals**: Keywords like "pass", "high grade", "balance"

#### Study Plan Generation (`generate_study_plan`)
**Pipeline:**
1. Retrieve relevant course info using `retrieval_service`
2. Build specialized prompt with constraints and course context
3. Call Ollama with structured prompt
4. Return formatted study plan

**Prompt Structure:**
```
STUDENT CONSTRAINTS:
- Available Time: X hours/week
- Goals: [goals]
- Workload: [preference]
- Days: [preferred days]

RELEVANT COURSE INFORMATION:
[Retrieved snippets]

STUDY PLAN REQUIREMENTS:
1. Weekly schedule with time allocations
2. Balance workload across days
3. Include lectures, self-study, assignments
4. Course-specific recommendations
5. Study tips based on goals

FORMAT:
## Weekly Study Schedule
## Course-Specific Recommendations
## Study Tips
## Weekly Hour Breakdown
```

### Usage
```python
from src.study_plan.manager import StudyPlanManager
from src.retrieval.lexical import lexical_search

# Initialize
manager = StudyPlanManager(retrieval_service=lexical_search)

# Start flow
response = manager.start_study_plan_flow()

# Collect constraints
result = manager.collect_constraint("I can study 15 hours per week")

# Generate plan
plan = manager.generate_study_plan(snippets, ollama_client)
```

### Integration in CLI
- Detects study plan queries: "study plan", "schedule", "weekly plan"
- Multi-turn dialogue for constraint collection
- Generates plan when constraints are complete
- Stores plan in conversation history

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

### Integration in `Module2_LexicalRetrieval.py`

```python
# Load data
snippets = load_snippets("./output/snippets.json")

# Lexical retrieval + RAG
answer = answer_with_lexical_rag(query, snippets)
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
│   │   └── service.py                    # Neural Retrieval
│   ├── conversation.py                   # Conversation Manager
│   ├── study_plan/
│   │   ├── __init__.py
│   │   └── manager.py                    # Study Plan Feature
│   └── ...
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

All components are now implemented and integrated into the HKBU Course Assistant system.
