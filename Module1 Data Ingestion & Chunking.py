#!pip install transformers

import os
import json
from transformers import GPT2Tokenizer

# Initialize the token counter (fixed code, no changes needed)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

'''Read All txt.files'''

def load_documents(folder_path):
    """
    Reads all .txt files in the specified folder and returns a list of documents
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process TXT files
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

# Call the function to read all TXT files in your course_docs folder
docs = load_documents("./course_docs")  # Relative path, corresponds to your Module1/course_docs
print(f"\nTotal documents read: {len(docs)}")

'''Split Long Documents into Small Snippets'''

def chunk_documents(docs, chunk_size=512, overlap=50):
    """
    Splits long documents into 512-token snippets with 50-token overlap (to avoid cutting off information)
    """
    snippets = []
    chunk_id = 0  # Assign a unique ID to each snippet
    
    for doc in docs:
        # Convert text to computer-readable tokens
        tokens = tokenizer.encode(doc["text"])
        # Split using a sliding window of chunk_size
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Add metadata (filename + ID) to the snippet
            snippets.append({
                "file_name": doc["file_name"],
                "chunk_id": chunk_id,
                "text": chunk_text.strip()  # Remove leading/trailing spaces
            })
            chunk_id += 1
    return snippets

# Call the chunking function
snippets = chunk_documents(docs)
print(f"Documents split into {len(snippets)} small snippets")

# Preview the first snippet (check if the format is correct)
if snippets:
    print("\nPreview of the first snippet:")
    print(f"Filename: {snippets[0]['file_name']}")
    print(f"Chunk ID: {snippets[0]['chunk_id']}")
    print(f"First 100 characters of content: {snippets[0]['text'][:100]}...")
    
    
'''Save Snippets to JSON & Write Loading Function'''
def save_snippets(snippets, save_path="./output/snippets.json"):
    """
    Saves the split snippets to a JSON file (for use by other team members)
    """
    # Ensure the output folder exists (auto-create if not)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save as JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(snippets, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(snippets)} snippets to: {save_path}")

def load_snippets(load_path="./output/snippets.json"):
    """
    Loads snippets from the JSON file (will be called by other team members)
    """
    try:
        with open(load_path, "r", encoding="utf-8") as f:
            snippets = json.load(f)
        print(f"Successfully loaded {len(snippets)} snippets")
        return snippets
    except Exception as e:
        print(f"Failed to load: {e}")
        return []

# 1. Save snippets to output/snippets.json
save_snippets(snippets)

# 2. Test the loading function (verify the file can be read normally)
loaded_snippets = load_snippets()

'''Final Validation'''
# Final validation: Call load_snippets to check format and content
final_snippets = load_snippets()

if final_snippets:
    print("Module1 Validation Passed!")
    print(f"Total snippets: {len(final_snippets)}")
    print(f"Source of first snippet: {final_snippets[0]['file_name']}")
    print(f"ID of first snippet: {final_snippets[0]['chunk_id']}")
    print(f"Preview of first snippet content:\n{final_snippets[0]['text'][:200]}...")
else:
    print("Validation failed, please check previous steps")
    