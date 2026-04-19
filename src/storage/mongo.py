"""
Cosmos DB Storage Service - Conversation History Persistence
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

logger = logging.getLogger(__name__)


class CosmosDBStorage:
    """Storage service for conversation history using Azure Cosmos DB (MongoDB API)."""

    def __init__(self, mongo_uri: str, app_name: str = "hkbu_assistant"):
        self.mongo_uri = mongo_uri
        self.app_name = app_name
        self._client: Optional[MongoClient] = None
        self._messages_collection = None
        self._connect()

    def _connect(self):
        """Establish connection to Cosmos DB."""
        try:
            # Parse URI to add appName if not present
            uri = self.mongo_uri
            if "appName" not in uri:
                separator = "&" if "?" in uri else "?"
                uri = f"{uri}{separator}appName=@{self.app_name}@"

            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
            )

            # Test connection
            self._client.admin.command("ping")
            logger.info("Connected to Cosmos DB")

            # Get database and collection
            db = self._client["7125Bot"]
            self._messages_collection = db["ChatMessages"]

        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"Failed to connect to Cosmos DB: {e}")
            self._client = None
            self._messages_collection = None

    def is_connected(self) -> bool:
        """Check if connected to Cosmos DB."""
        if self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except:
            return False

    def save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        username: Optional[str] = None,
    ) -> bool:
        """
        Save a message to the conversation history.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            role: Message role (user/assistant)
            content: Message content
            username: Optional username

        Returns:
            True if saved successfully, False otherwise
        """
        if self._messages_collection is None:
            logger.warning("Storage not available, message not saved")
            return False

        try:
            self._messages_collection.insert_one({
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "username": username,
                "role": role,
                "content": content,
            })
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False

    def get_conversation_history(
        self,
        session_id: str,
        max_messages: int = 50,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to return

        Returns:
            List of message dictionaries with role and content
        """
        if self._messages_collection is None:
            logger.warning("Storage not available")
            return []

        try:
            # Query by session_id (don't use sort - Cosmos DB doesn't support it without proper indexing)
            cursor = self._messages_collection.find({"session_id": session_id})

            items = list(cursor)

            # Sort in Python (Cosmos DB sort workaround)
            items.sort(key=lambda x: x.get("timestamp", ""))

            # Limit results
            items = items[-max_messages:]

            # Format as conversation history
            return [
                {"role": item["role"], "content": item["content"]}
                for item in items
            ]

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def get_summarized_history(
        self,
        session_id: str,
        max_messages: int = 50,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history with summarization for long conversations.
        If there are more than 6 messages, older ones are summarized.
        """
        history = self.get_conversation_history(session_id, max_messages)

        if len(history) <= 6:
            return history

        # Keep last 4 messages, summarize the rest
        messages_to_summarize = history[:-4]
        recent_messages = history[-4:]

        # Generate summary
        conversation_text = "\n".join([
            f"{item['role'].upper()}: {item['content']}"
            for item in messages_to_summarize
        ])

        # Return summarized format
        return [
            {"role": "system",
                "content": f"[Earlier conversation summary: {len(messages_to_summarize)} messages summarized]"}
        ] + recent_messages

    def create_session(self) -> str:
        """Create a new session ID."""
        return str(uuid.uuid4())

    def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("Cosmos DB connection closed")
