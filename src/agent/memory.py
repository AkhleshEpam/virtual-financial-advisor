"""
memory.py — Session memory management for the financial advisor agent.
Uses LangChain ChatMessageHistory for multi-turn interactions.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class SessionMemory:
    """Manages conversation history and cached user profile for one session."""

    def __init__(self):
        self._history = ChatMessageHistory()
        self._user_profile: dict | None = None

    def save_context(self, user_input: str, agent_output: str) -> None:
        """Store a single interaction turn."""
        self._history.add_user_message(user_input)
        self._history.add_ai_message(agent_output)

    def get_history(self) -> list:
        """Return the full conversation message history."""
        return self._history.messages

    def get_history_str(self) -> str:
        """Return conversation history as a formatted string."""
        messages = self.get_history()
        lines = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Advisor"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset conversation history and cached profile."""
        self._history.clear()
        self._user_profile = None

    def set_user_profile(self, profile: dict) -> None:
        """Cache the user's financial profile for the session."""
        self._user_profile = profile

    def get_user_profile(self) -> dict | None:
        """Return the cached user financial profile, or None if not loaded."""
        return self._user_profile

    @property
    def chat_history(self) -> ChatMessageHistory:
        """Expose the underlying ChatMessageHistory for agent integration."""
        return self._history
