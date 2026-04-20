"""
CLI Interface - Main entry point for HKBU Course Assistant
"""
from src.storage.mongo import CosmosDBStorage
from src.ollama.chat import OllamaChatService
from src.rag.service import RAGService
from src.config import load_config, AppConfig
from src.conversation import ConversationManager
from src.study_plan.manager import StudyPlanManager
import os
import sys
import logging
import uuid
import json
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Import lexical search
try:
    from Module2_LexicalRetrieval import lexical_search
except ImportError:
    lexical_search = None


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
        self.study_plan_manager: Optional[StudyPlanManager] = None
        self.snippets: list = []

        # Initialize conversation manager for multi-turn Q&A
        self.conversation_manager: Optional[ConversationManager] = None

        self._initialize_services()

    def _hydrate_conversation_history(self):
        if not self.storage or not self.session_id or not self.conversation_manager:
            return

        try:
            history = self.storage.get_conversation_history(self.session_id)
        except Exception as e:
            logger.warning(f"Failed to hydrate conversation history: {e}")
            return

        for item in history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role == "user" and isinstance(content, str):
                self.conversation_manager.add_user_message(content)
            elif role == "assistant" and isinstance(content, str):
                self.conversation_manager.add_assistant_message(content)

    def _initialize_services(self):
        """Initialize all services with error handling."""
        # Initialize RAG Service
        try:
            logger.info("Initializing RAG service...")
            self.rag_service = RAGService(
                data_dir=self.config.rag.data_dir,
                chroma_path=self.config.rag.chroma_path,
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
                rebuild_if_changed=self.config.rag.rebuild_if_changed,
                ollama_base_url=self.config.ollama.base_url,
                ollama_embed_model=self.config.rag.ollama_embed_model,
            )
            logger.info(
                f"RAG ready with {self.rag_service.count()} chunks")
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
        resume_session_id = os.getenv("HKBU_SESSION_ID")
        if isinstance(resume_session_id, str) and resume_session_id.strip():
            self.session_id = resume_session_id.strip()
        elif self.storage:
            self.session_id = self.storage.create_session()
        else:
            self.session_id = str(uuid.uuid4())

        # Initialize Conversation Manager for multi-turn Q&A
        try:
            logger.info("Initializing Conversation Manager...")
            self.conversation_manager = ConversationManager(
                system_message=SYSTEM_PROMPT,
                session_id=self.session_id,
                max_turns=6,  # Keep last 6 turns (12 messages) for context
            )
            self._hydrate_conversation_history()
            logger.info("Conversation Manager ready")
        except Exception as e:
            logger.warning(f"Conversation Manager initialization failed: {e}")
            self.conversation_manager = None

        # Initialize Study Plan Manager
        try:
            logger.info("Initializing Study Plan Manager...")
            self.study_plan_manager = StudyPlanManager(
                retrieval_service=lexical_search
            )
            # Load snippets for study plan retrieval
            self.snippets = self._load_snippets()
            logger.info(
                f"Study Plan Manager ready with {len(self.snippets)} snippets")
        except Exception as e:
            logger.warning(f"Study Plan Manager initialization failed: {e}")
            self.study_plan_manager = None

    def _load_snippets(self) -> list:
        """Load snippets from JSON for study plan retrieval."""
        try:
            with open("./output/snippets.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load snippets: {e}")
            return []

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
        Uses ConversationManager for multi-turn Q&A with automatic history truncation.
        """
        if not self.chat_service:
            return "Error: Chat service not available. Please ensure Ollama is running."

        # Check if this is a study plan query
        if self.study_plan_manager and self.study_plan_manager.is_study_plan_query(user_message):
            return self._handle_study_plan_query(user_message)

        # Add user message to conversation manager
        if self.conversation_manager:
            self.conversation_manager.add_user_message(user_message)

        # Get RAG context
        rag_context = self._get_rag_context(user_message)

        # Build system prompt with context
        system_prompt = SYSTEM_PROMPT
        if rag_context:
            system_prompt += rag_context

        # Get conversation history from ConversationManager (with max_turns truncation)
        if self.conversation_manager:
            history = self.conversation_manager.get_history()
        else:
            # Fallback to storage-based history
            history = self._get_conversation_history()
        if history:
            history = [m for m in history if m.get("role") != "system"]

        # Generate response
        response = self.chat_service.chat(
            message=user_message,
            conversation_history=history,
            system_prompt=system_prompt,
        )

        # Add assistant response to conversation manager
        if self.conversation_manager:
            self.conversation_manager.add_assistant_message(response)

        # Save interaction to persistent storage
        self._save_interaction(user_message, response)

        return response

    def _handle_study_plan_query(self, user_message: str) -> str:
        """Handle study plan generation workflow."""
        if not self.study_plan_manager:
            return "Study Plan feature is not available."

        state = self.study_plan_manager.conversation_state

        # Start new study plan flow
        if state == "collecting_constraints" and self.study_plan_manager.user_constraints is None:
            response = self.study_plan_manager.start_study_plan_flow()
            self._save_interaction(user_message, response)
            return response

        # Collect constraints
        if state in ["collecting_constraints", "ready_to_generate"]:
            result = self.study_plan_manager.collect_constraint(user_message)

            if result["status"] == "ready":
                # Ask for confirmation before generating
                response = result["message"]
            elif result["status"] == "collecting":
                response = result["message"]
            else:
                response = "Let's continue. " + result["message"]

            self._save_interaction(user_message, response)
            return response

        # Generate plan when user confirms
        if state == "ready_to_generate" and user_message.lower() in ["yes", "y", "sure", "ok"]:
            print(
                "\nGenerating your personalized study plan... This may take a moment.\n")
            result = self.study_plan_manager.generate_study_plan(
                snippets=self.snippets,
                ollama_client=self.chat_service,
            )

            if result["status"] == "success":
                response = f"## Your Personalized Study Plan\n\n{result['study_plan']}"
                # Reset for next time
                self.study_plan_manager.reset()
            else:
                response = f"Sorry, I couldn't generate the study plan: {result['message']}"

            self._save_interaction(user_message, response)
            return response

        # Default: treat as regular message
        return self._continue_study_plan_conversation(user_message)

    def _continue_study_plan_conversation(self, user_message: str) -> str:
        """Continue study plan constraint collection."""
        result = self.study_plan_manager.collect_constraint(user_message)
        self._save_interaction(user_message, result["message"])
        return result["message"]

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
        print(
            f"Study Plan: {'Enabled' if self.study_plan_manager else 'Disabled'}")
        print("-" * 60)
        print("Type your questions or 'exit' to quit")
        print("Try: 'study plan' to generate a personalized study schedule\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "new":
                    if self.storage:
                        self.session_id = self.storage.create_session()
                    else:
                        self.session_id = str(uuid.uuid4())
                    self.conversation_manager = ConversationManager(
                        system_message=SYSTEM_PROMPT,
                        session_id=self.session_id,
                        max_turns=6,
                    )
                    print(f"New session started: {self.session_id}")
                    continue

                if user_input.lower() == "help":
                    print("\nCommands:")
                    print("  exit/quit - Exit the program")
                    print("  new - Start a new session")
                    print("  help - Show this help message")
                    print("  study plan - Generate a personalized study plan")
                    print()
                    continue

                if user_input.lower() == "study plan":
                    user_input = "I want to create a study plan"

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
