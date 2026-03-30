"""
Simple Main Entry Point - HKBU Course Assistant
"""
from src.storage.mongo import CosmosDBStorage
from src.ollama.chat import OllamaChatService
from src.rag.service import RAGService
from src.config import load_config
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are BU Assistant, an intelligent campus assistant for HKBU (Hong Kong Baptist University).
Your role is to help students and visitors with questions about the university, including:
- Academic programs and courses
- Campus facilities and locations
- Student life and services
- University policies and procedures
- General information about HKBU

Please answer questions based on the provided context from the HKBU knowledge base.
If you don't have enough information to answer a question, politely indicate that
and suggest where they might find the information."""


def main():
    """Simple main function to test the assistant."""
    # Load configuration
    config = load_config()

    print("=" * 60)
    print("HKBU Course Assistant")
    print("=" * 60)

    # Initialize RAG Service
    print("\nInitializing RAG service...")
    try:
        rag_service = RAGService(
            data_dir="./course_docs",
            chroma_path="./chroma_db",
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        print(f"RAG ready: {rag_service.collection.count()} chunks")
    except Exception as e:
        print(f"RAG failed: {e}")
        rag_service = None

    # Initialize Ollama
    print("\nInitializing Ollama...")
    try:
        chat_service = OllamaChatService(
            model=config.ollama.model,
            base_url=config.ollama.base_url,
        )
        if chat_service.is_available():
            print(f"Ollama available: {config.ollama.model}")
        else:
            print("Ollama not available. Please run: ollama serve")
            return
    except Exception as e:
        print(f"Ollama failed: {e}")
        return

    # Initialize Storage (optional)
    print("\nInitializing Cosmos DB...")
    try:
        storage = CosmosDBStorage(
            mongo_uri=config.mongo.uri,
            app_name="hkbu_assistant",
        )
        if storage.is_connected():
            print("Cosmos DB connected")
        else:
            print("Cosmos DB not connected (continuing without)")
            storage = None
    except Exception as e:
        print(f"Cosmos DB failed: {e} (continuing without)")
        storage = None

    # Create session
    session_id = storage.create_session() if storage else "test_session"
    print(f"\nSession ID: {session_id}")
    print("-" * 60)

    # Test query
    test_query = "What academic programs does HKBU offer?"
    print(f"\nTest Query: {test_query}\n")

    # Get RAG context
    if rag_service:
        context = rag_service.get_context(test_query, k=config.rag.top_k)
        if context:
            print(f"RAG Context retrieved: {len(context)} characters")
        else:
            print("No RAG context found")
    else:
        context = None

    # Build system prompt
    system_prompt = SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\n[HKBU Campus Knowledge]\n{context}\n[End of Campus Knowledge]"

    # Get response
    print("\nGenerating response...\n")
    response = chat_service.chat(
        message=test_query,
        system_prompt=system_prompt,
    )

    print(f"Assistant: {response}\n")

    # Save to storage if available
    if storage:
        storage.save_message(session_id, "test_user", "user", test_query)
        storage.save_message(session_id, "test_user", "assistant", response)
        print("Conversation saved to Cosmos DB")

    print("=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
