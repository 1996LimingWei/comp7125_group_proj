"""
Ollama Chat Service - Chat API wrapper for local LLM
"""
import json
import logging
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaChatService:
    """Chat service using Ollama's Chat API."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_endpoint = f"{base_url}/api/chat"

    def chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a chat message to Ollama and get a response.

        Args:
            message: User's message
            conversation_history: Previous messages in format [{"role": "user|assistant", "content": "..."}]
            system_prompt: System prompt for context

        Returns:
            Assistant's response text
        """
        # Build messages list
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({
            "role": "user",
            "content": message,
        })

        # Prepare payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            return result["message"]["content"]

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is Ollama running?")
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running."
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Error: Request timed out. Please try again."
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except:
            return False

    def get_model_info(self) -> Dict:
        """Get information about available models."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
