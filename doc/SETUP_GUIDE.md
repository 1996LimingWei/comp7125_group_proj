# HKBU Course Assistant - Setup & Testing Guide

## Prerequisites

1. **Python 3.11+** installed
2. **Ollama** installed and running

---

## Quick Start

### 1. Install Ollama and Pull Model

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# In a new terminal, pull the model
ollama pull gemma3:4b
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

```bash
python "Module1 Data Ingestion & Chunking.py"
```

This creates `./output/snippets.json` with chunked course documents.

### 4. Run the Application

**Simple Test (Recommended for first run):**
```bash
python main.py
```

**Interactive CLI Mode:**
```bash
python -m src.cli.main
```

### 5. Test Features Individually
**Test Lexical Retrieval:**
```bash
python Module2_LexicalRetrieval.py
```

**Test Study Plan Feature:**
```bash
python test_study_plan.py
```

---

## Testing the Application

### Interactive CLI Commands

Once running, you can use these commands:
- `exit` / `quit` - Exit the program
- `new` - Start a new conversation session
- `help` - Show available commands

### Step-by-Step Test Flow

Test each module in the integrated CLI:

#### 1. Test Neural RAG
```
You: What academic programs does HKBU offer?
```
Expected: Answer using vector search from ChromaDB with citations

#### 2. Test Conversation Memory
```
You: Tell me more about that
```
Expected: Maintains context from previous question

#### 3. Test Study Plan Feature
```
You: study plan
Assistant: I'll help you create a personalized study plan!...
You: 15 hours per week
You: I want to pass all courses
You: moderate workload
```
Expected: Multi-turn dialogue collecting constraints, then generates study plan

#### 4. Test Regular Query (Non-Study Plan)
```
You: What is the tuition fee?
```
Expected: Regular RAG answer (not study plan flow)

#### 5. Test New Session
```
You: new
```
Expected: Starts fresh conversation with new session ID

#### 6. Exit
```
You: exit
```

---

### Module Testing Summary

| Component | Test Command | What It Tests |
|-----------|-------------|---------------|
| Data Ingestion | `python "Module1 Data Ingestion & Chunking.py"` | Data chunking |
| Lexical Retrieval | `python Module2_LexicalRetrieval.py` | BM25 lexical search |
| Neural Retrieval | `python main.py` | Neural RAG with ChromaDB |
| Conversation Manager | `python -m src.cli.main` → ask follow-up questions | Conversation history |
| Study Plan Feature | `python test_study_plan.py` | Study plan generation |
| All Components | `python -m src.cli.main` | Full integration |