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

### 3. Run the Application

**Simple Test (Recommended for first run):**
```bash
python main.py
```

**Interactive CLI Mode:**
```bash
python -m src.cli.main
```

---

## Testing the Application

### Interactive CLI Commands

Once running, you can use these commands:
- `exit` / `quit` - Exit the program
- `new` - Start a new conversation session
- `help` - Show available commands

### Sample Test Queries

Try these queries to verify the system is working:

1. **Academic Information:**
   ```
   What academic programs does HKBU offer?
   ```

2. **Campus Information:**
   ```
   Tell me about student life at HKBU
   ```

3. **Policy Questions:**
   ```
   What is the academic integrity policy?
   ```

4. **Fee Information:**
   ```
   What are the tuition fees for international students?
   ```