# HKBU Course Assistant - Setup & Testing Guide

## Prerequisites

1. **Python 3.11+** installed
2. **Node.js 18+** installed (for Web UI)
3. **Ollama** installed and running

---

## Quick Start

### 1. Install Ollama and Pull Models

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# In a new terminal, pull the required models
ollama pull gemma3:4b
ollama pull nomic-embed-text
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
- **Generation Controls**: Adjust temperature and max_tokens via settings panel

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
- `study plan` - Start the study plan workflow

### Web UI Features

The web interface provides a modern ChatGPT-like experience:

- **Sidebar**: New chat, quick actions, session management
- **Welcome Screen**: Suggested queries to get started
- **Chat Interface**: Real-time messaging with markdown support
- **Study Plan**: Interactive constraint collection dialog
- **Generation Controls**: Temperature and max_tokens sliders
- **Responsive Design**: Works on desktop and mobile

### Sample Interaction: COMP7125 Study Plan

Try this complete example in CLI or Web UI:

**You:**
```
I am study COMP7125 Prompt Engineering for Generative AI as an itm student. I am struggled with this course, please help me make a study plan.
```

**Assistant:**
```
I'll help you create a personalized study plan!

To get started, please tell me:
1. How many hours per week can you dedicate to studying?
2. What are your main goals? (e.g., 'pass all courses', 'achieve high grades', 'balance work and study')
3. What's your preferred workload? (light/moderate/heavy)
4. Which days of the week work best for you?
5. Any specific courses you want to focus on?
```

**You:**
```
20 hours per week, high grade, moderate workload, Monday Wednesday Friday
```

**Assistant:**
```
Great! Here's what I've collected:

📅 Available Hours: 20 hours/week
🎯 Goals: high grade
⚡ Workload Preference: moderate
📆 Preferred Days: Monday, Wednesday, Friday
📚 Focus Courses: COMP7125

Shall I generate your study plan now? (yes/no)
```

**You:**
```
yes
```

**Assistant:** Generates a personalized weekly study schedule for COMP7125 based on the course outline, including time allocations for lectures, self-study, assignments, and revision.

#### 4. Test Generation Controls
In the Web UI settings panel:
- Adjust temperature slider (0.3 for factual, 0.7 for creative)
- Adjust max_tokens slider (512 for short, 1024 for detailed)

Expected: Responses adapt to parameter changes


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