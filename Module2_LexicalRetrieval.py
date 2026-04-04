# Module1: Data Ingestion & Chunking
# Pipeline: Read TXT → Chunk → Save JSON → Load & Validate

# Install dependencies (uncomment if needed)
#!pip install transformers

import os
import json
from transformers import GPT2Tokenizer

# Initialize tokenizer (fixed, no modification)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def load_documents(folder_path):
    """
    Read all .txt files from the specified folder.
    Return a list of documents with filename and text content.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append({
                        "file_name": filename,
                        "text": text
                    })
                print(f"Successfully read: {filename}")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    return docs

def chunk_documents(docs, chunk_size=512, overlap=50):
    """
    Split long documents into fixed-length token chunks with overlap.
    Return structured snippets with file name, chunk ID, and chunk text.
    """
    snippets = []
    chunk_id = 0

    for doc in docs:
        tokens = tokenizer.encode(doc["text"])
        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)

            snippets.append({
                "file_name": doc["file_name"],
                "chunk_id": chunk_id,
                "text": chunk_text.strip()
            })
            chunk_id += 1
    return snippets

def save_snippets(snippets, save_path="./output/snippets.json"):
    """
    Save chunked snippets to JSON for later use by Module2.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(snippets, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(snippets)} snippets to: {save_path}")

def load_snippets(load_path="./output/snippets.json"):
    """
    Load snippets from JSON (used by teammates & Module2).
    """
    try:
        with open(load_path, "r", encoding="utf-8") as f:
            snippets = json.load(f)
        print(f"Successfully loaded {len(snippets)} snippets")
        return snippets
    except Exception as e:
        print(f"Failed to load: {e}")
        return []

# ------------------------------
# Main execution flow
# ------------------------------
if __name__ == "__main__":
    # 1. Load all txt files
    docs = load_documents("./course_docs")
    print(f"\nTotal documents read: {len(docs)}")

    # 2. Split into chunks
    snippets = chunk_documents(docs)
    print(f"Documents split into {len(snippets)} small snippets")

    # Preview first snippet
    if snippets:
        print("\nPreview of the first snippet:")
        print(f"Filename: {snippets[0]['file_name']}")
        print(f"Chunk ID: {snippets[0]['chunk_id']}")
        print(f"First 100 characters: {snippets[0]['text'][:100]}...\n")

    # 3. Save & reload to verify
    save_snippets(snippets)
    loaded_snippets = load_snippets()

    # 4. Final validation
    final_snippets = load_snippets()
    if final_snippets:
        print("=" * 50)
        print("Module1 Validation Passed!")
        print(f"Total snippets: {len(final_snippets)}")
        print(f"First snippet source: {final_snippets[0]['file_name']}")
        print(f"First snippet ID: {final_snippets[0]['chunk_id']}")
        print(f"Content preview:\n{final_snippets[0]['text'][:200]}...")
        print("=" * 50)
    else:
        print("Module1 Validation Failed!")