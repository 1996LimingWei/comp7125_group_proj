"""
CLI Interface - Main entry point for HKBU Course Assistant
"""
from src.storage.mongo import CosmosDBStorage
from src.ollama.chat import OllamaChatService
from src.rag.service import RAGService
from src.config import load_config, AppConfig
import os
import sys
import logging
import uuid
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


class HKBUAssistant:
    """Main application class for HKBU Course Assistant."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.user_id = "cli_user"
        self.session_id: Optional[str] = None

        # Initialize services with graceful degradation
        self.rag_service: Optional[RAGService] = None
        self.chat_service: Optional[OllamaChatService] = None
        self.storage: Optional[CosmosDBStorage] = None

        self._initialize_services()

    def _initialize_services(self):
        """Initialize all services with error handling."""
        # Initialize RAG Service
        try:
            logger.info("Initializing RAG service...")
            self.rag_service = RAGService(
                data_dir="./course_docs",
                chroma_path=self.config.rag.chroma_path,
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
            )
            logger.info(
                f"RAG ready with {self.rag_service.collection.count()} chunks")
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            self.rag_service = None

        # Initialize Ollama Chat Service
        try:
            logger.info("Initializing Ollama chat service...")
            self.chat_service = OllamaChatService(
                model=self.config.ollama.model,
                base_url=self.config.ollama.base_url,
            )
            if self.chat_service.is_available():
                logger.info("Ollama is available")
            else:
                logger.warning("Ollama is not available")
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
            self.chat_service = None

        # Initialize Cosmos DB Storage
        try:
            logger.info("Initializing Cosmos DB storage...")
            self.storage = CosmosDBStorage(
                mongo_uri=self.config.mongo.uri,
                app_name="hkbu_assistant",
            )
            if self.storage.is_connected():
                logger.info("Cosmos DB connected")
            else:
                logger.warning("Cosmos DB not connected")
        except Exception as e:
            logger.warning(f"Cosmos DB initialization failed: {e}")
            self.storage = None

        # Create new session
        if self.storage:
            self.session_id = self.storage.create_session()
        else:
            self.session_id = str(uuid.uuid4())

    def _get_rag_context(self, query: str) -> str:
        """Get RAG context for the query."""
        if not self.rag_service:
            return ""

        context = self.rag_service.get_context(
            query,
            k=self.config.rag.top_k,
        )

        if context:
            return (
                "\n\n[HKBU Campus Knowledge - use this to answer the user]\n"
                + context +
                "\n[End of Campus Knowledge]"
            )
        return ""

    def _get_conversation_history(self) -> list:
        """Get conversation history from storage."""
        if not self.storage or not self.session_id:
            return []
        return self.storage.get_conversation_history(self.session_id)

    def _save_interaction(self, user_message: str, assistant_response: str):
        """Save the interaction to storage."""
        if not self.storage or not self.session_id:
            return

        self.storage.save_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role="user",
            content=user_message,
        )
        self.storage.save_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role="assistant",
            content=assistant_response,
        )

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return an assistant response.
        """
        if not self.chat_service:
            return "Error: Chat service not available. Please ensure Ollama is running."

        # Get RAG context
        rag_context = self._get_rag_context(user_message)

        # Build system prompt with context
        system_prompt = SYSTEM_PROMPT
        if rag_context:
            system_prompt += rag_context

        # Get conversation history
        history = self._get_conversation_history()

        # Generate response
        response = self.chat_service.chat(
            message=user_message,
            conversation_history=history,
            system_prompt=system_prompt,
        )

        # Save interaction
        self._save_interaction(user_message, response)

        return response

    def run_interactive(self):
        """Run the interactive CLI chat loop."""
        print("\n" + "=" * 60)
        print("HKBU Course Assistant - Interactive Mode")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"RAG: {'Enabled' if self.rag_service else 'Disabled'}")
        print(
            f"Ollama: {'Available' if self.chat_service and self.chat_service.is_available() else 'Not Available'}")
        print(
            f"Storage: {'Connected' if self.storage and self.storage.is_connected() else 'Not Connected'}")
        print("-" * 60)
        print("Type your questions or 'exit' to quit\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "new":
                    self.session_id = str(uuid.uuid4())
                    print(f"New session started: {self.session_id}")
                    continue

                if user_input.lower() == "help":
                    print("\nCommands:")
                    print("  exit/quit - Exit the program")
                    print("  new - Start a new session")
                    print("  help - Show this help message")
                    print()
                    continue

                # Process the message
                response = self.chat(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}\n")


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()

    # Create and run assistant
    assistant = HKBUAssistant(config)
    assistant.run_interactive()


if __name__ == "__main__":
    main()
