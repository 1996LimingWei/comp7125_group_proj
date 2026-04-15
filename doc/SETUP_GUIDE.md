# HKBU Course Assistant - Setup & Testing Guide

## Prerequisites

1. **Python 3.11+** installed
2. **Node.js 18+** installed (for Web UI)
3. **Ollama** installed and running

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

**Option 1: Web UI (Recommended - Modern Interface)**

The web UI provides a ChatGPT-like interface with a modern design, sidebar navigation, and real-time chat.

**Step 1: Start the Backend (Port 5001)**
```bash
conda activate 7125-hkbu
python web_app.py
```
You should see:
```
HKBU Assistant Web Server (CLI Class Wrapper)
This server uses the CLI's HKBUAssistant class directly.
Starting server on http://localhost:5001
```

**How it works:**
- Uses the CLI's `HKBUAssistant` class directly
- Guarantees 100% identical behavior to CLI
- No code duplication or divergent logic
- Supports all CLI features including study plan

**Step 2: Start the Frontend (Port 3000)**
```bash
cd web_ui
npm install  # First time only
npm start
```
You should see:
- Compiled successfully!
- Local: http://localhost:3000

**Step 3: Open Browser**
Navigate to http://localhost:3000

**Features:**
- **Sidebar**: New chat button, quick actions (Study Plan, Programs), session info
- **Welcome Screen**: Click suggestion cards or type your question
- **Chat Interface**: Markdown support, code blocks, typing indicators
- **Study Plan**: Click "Study Plan" quick action to start constraint collection

**Option 2: CLI Mode**
```bash
conda activate 7125-hkbu
python -m src.cli.main
```

**Option 3: Simple Test**
```bash
conda activate 7125-hkbu
python main.py
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

### Web UI Features

The web interface provides a modern ChatGPT-like experience:

- **Sidebar**: New chat, quick actions, session management
- **Welcome Screen**: Suggested queries to get started
- **Chat Interface**: Real-time messaging with markdown support
- **Study Plan**: Interactive constraint collection dialog
- **Responsive Design**: Works on desktop and mobile

### Step-by-Step Test Flow (Web UI)

#### 1. Test Neural RAG
Type in the chat:
```
What academic programs does HKBU offer?
```
Expected: Answer using vector search from ChromaDB with citations

#### 2. Test Conversation Memory
Follow up with:
```
Tell me more about that
```
Expected: Maintains context from previous question

#### 3. Test Study Plan Feature
Click "Study Plan" quick action or type:
```
Create a study plan
```
Then respond to the prompts:
- Hours per week: `15`
- Goals: `I want to pass all courses`
- Workload: `moderate`

Expected: Multi-turn dialogue collecting constraints, then generates study plan

#### 4. Test Regular Query
Type:
```
What is the tuition fee?
```
Expected: Regular RAG answer (not study plan flow)

#### 5. Test New Session
Click "New Chat" button in sidebar
Expected: Starts fresh conversation with new session ID

### Step-by-Step Test Flow (CLI)

If using CLI mode (`python -m src.cli.main`):

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