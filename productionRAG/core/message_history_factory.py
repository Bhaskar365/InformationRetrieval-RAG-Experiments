

"""
Message History Factory Pattern Implementation.
Supports seamless switching between in-memory (development) and Redis (production) storage.
Demonstrates session management and state persistence patterns.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import json
import uuid

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


from langchain_community.chat_message_histories import InMemoryChatMessageHistory

try:
    from langchain_community.chat_message_histories import RedisChatMessageHistory
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisChatMessageHistory = None

from config.settings import settings

class BaseMessageHistory(ABC):
    """Abstract base for message history implementations."""

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to history."""
        pass


    @abstractmethod
    def get_messages(self) -> List[BaseMessage]:
        """Retrieve all messages."""
        pass


    @abstractmethod
    def clear(self) -> None:
        """Clear all messages"""
        pass


class MemoryMessageHistory(BaseMessageHistory):
    """
    In-memory message history.
    Fast, zero dependencies, but data is lost on restart.
    Ideal for development and single-session demos.
    """

    def __init__(self, session_id:str):
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
        print(f"In-Memory History initialized for session: {session_id[:8]}...")

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        return self._messages.copy()

    def clear(self) -> None:
        self._messages = []
        print(f"🗑️  Memory cleared for session: {self.session_id[:8]}...")

    def to_langchain_history(self) -> BaseChatMessageHistory:
        """Return LangChain-compatible history object."""
        return InMemoryChatMessageHistory(messages=self._messages)



class RedisMessageHistory(BaseMessageHistory):
    """
    Redis-backed message history.
    Persistent, distributed, scalable.
    Ideal for production multi-user deployments.
    """
    
    def __init__(self, session_id: str, user_id: str = "anonymous"):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis packages not installed. "
                "Run: pip install redis langchain-community"
            )
        
        self.session_id = session_id
        self.user_id = user_id
        self.config = settings.message_history
        
        # Secure key design: user-scoped sessions
        self.key = f"{self.config.redis_key_prefix}:{user_id}:{session_id}"
        
        print(f"🔌 Redis History initialized: {self.key}")
        
        self._history = RedisChatMessageHistory(
            session_id=self.key,
            url=self.config.redis_url,
            ttl=self.config.redis_ttl
        )
    
    def add_message(self, message: BaseMessage) -> None:
        self._history.add_message(message)
    
    def get_messages(self) -> List[BaseMessage]:
        return self._history.messages
    
    def clear(self) -> None:
        self._history.clear()
        print(f"🗑️  Redis cleared: {self.key}")
    
    def to_langchain_history(self) -> BaseChatMessageHistory:
        """Return LangChain-compatible history object."""
        return self._history


class MessageHistoryFactory:
    """
    Factory for creating message history instances.
    Demonstrates session isolation and multi-tenancy patterns.
    """

    _histories: Dict[str, BaseMessageHistory] = {}

    @classmethod
    def create(cls, session_id: Optional[str]=None, user_id: str = "anonymous", history_type:Optional[str]=None)->BaseMessageHistory:
        """
        Create a message history instance.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
            user_id: User identifier for multi-tenant isolation
            history_type: "memory" or "redis" (defaults to settings)
        
        Returns:
            BaseMessageHistory implementation
        """

        history_type = history_type or settings.message_history.store_type
        session_id = session_id or str(uuid.uuid4)

        # create cache key
        cache_key = f"{history_type}:{user_id}:{session_id}"

        # Return cached instance if exists
        if cache_key in cls._histories:
            return cls._histories[cache_key]

        # create new instance
        if history_type == "redis":
            try:
                history = RedisMessageHistory(session_id, user_id)
                cls._histories[cache_key] = history
                return history
            except Exception as e:
                print(f"Redis connection failed: {e}")
                print("Falling back to In-Memory...")
                history_type = "memory"
        
        # Default to memory
        history = MemoryMessageHistory(session_id)
        cls._histories[cache_key] = history
        return history

    @classmethod
    def get_session_history(cls, session_id:str, user_id:str='Anonymous'):
        """
        Convenience method for LangChain's RunnableWithMessageHistory.
        Returns a LangChain-compatible history object.
        """

        history = cls.create(session_id, user_id)
        return history.to_langchain_history()

    @classmethod
    def clear_cache(cls):
        """Clear all cached histories."""
        cls._histories.clear()
        print("Message history cache cleared")


    @classmethod
    def list_active_sessions(cls) -> List[str]:
        """List all active session IDs (for monitoring)."""
        return list(cls._histories.keys())


# Convenience function
def get_message_history(session_id: Optional[str] = None,user_id: str = "anonymous") -> BaseMessageHistory:
    """Get configured message history instance."""
    return MessageHistoryFactory.create(session_id, user_id)

