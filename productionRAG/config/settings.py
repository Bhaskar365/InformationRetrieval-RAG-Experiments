

"""
Centralized configuration management with environment-based switching.
Demonstrates production-grade configuration patterns.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider:str = "ollama"        # "ollama" | "openrouter"
    api_key:Optional[str] = None
    generation_model: str = "llama3.2:3b"
    critic_model: str = "llama3.2:3b"
    temperature:float = 0.0
    base_url:Optional[str]=None

    def __post__init__(self):
        if self.provider == "openrouter":
            self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "ollama":
            self.base_url = self.base_url or "http://localhost:11434"

@dataclass
class VectorStoreConfig:
        """Vector store configuration with factory pattern support."""
        store_type:str = "faiss"        # "faiss" | "pinecone"

        # FAISS SETTINGS
        faiss_index_path:str = "./vector_db/faiss_index"

        # Pinecone settings
        pinecone_api_key:Optional[str] = None
        pinecone_environment:Optional[str] = None
        pinecone_index_name:str = "production-rag-index"
        pinecone_dimension:int = 384        # Match your embedding model
        pinecone_metric:str = 'cosine'
        pinecone_cloud:str = 'aws'
        pinecone_region:str = 'us-east-1'

        def __post_init__(self):
            self.pinecone_api_key = self.pinecone_api_key or os.getenv("PINECONE_API_KEY")
            self.pinecone_environment = self.pinecone_environment or os.getenv("PINECONE_ENVIRONMENT")


@dataclass
class MemoryHistoryConfig:
        """Message history configuration."""
        store_type:str = "memory"   # "memory" | "redis"

        # Redis settings
        redis_url:str = "redis://localhost:6379"
        redis_key_prefix:str = "rag:session"
        redis_ttl:int = 86400

        def __post__init(self):
            self.redis_url = os.getenv("REDIS_URL", self.redis_url)

@dataclass
class EmbeddingConfig:
        """Embedding model configuration."""
        model_name:str = "all-MiniLM-L6-v2"
        device:str = "cpu"      # "cpu" | "cuda"

class Settings:
            """
            Master settings class that aggregates all configuration.
            Demonstrates the Singleton pattern for global config access.
            """
            _instance = None

            def __new__(cls):
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
                return cls._instance

            def _initialize(self):
                self.llm = LLMConfig(
                    provider=os.getenv("LLM_PROVIDER", "ollama"),
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    generation_model=os.getenv("GENERATION_MODEL", "llama3.2:3b"),
                    critic_model=os.getenv("CRITIC_MODEL", "llama3.2:3b"),
                )


                self.vector_store = VectorStoreConfig(
                      store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),
                      pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                      pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
                )

                self.message_history = MemoryHistoryConfig(
                      store_type=os.getenv("HISTORY_STORE_TYPE","memory"),
                      redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
                )

                self.embedding = EmbeddingConfig(
                      model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                      device=os.getenv("EMBEDDING_DEVICE", "cpu")
                )

                # Derived settings
                self.debug = os.getenv("DEBUG", "false").lower() == "true"
                self.log_level = os.getenv("LOG_LEVEL", "INFO")

            def is_production_mode(self)->bool:
                """Check if running in production configuration."""
                return (
                     self.vector_store.store_type == "pinecone" or
                     self.message_history.store_type == "redis"
                )

            def validate(self) -> list[str]:
                """Validate configuration and return list of errors."""
                errors = []

                if self.llm.provider == "openrouter" and not self.llm.api_key:
                     errors.append("OPENROUTER_API_KEY required when LLM_PROVIDER=openrouter")

                if self.vector_store.store_type == "pinecone":
                    if not self.vector_store.pinecone_api_key:
                         errors.append("PINECONE_API_KEY required when VECTOR_STORE_TYPE=pinecone")

                    if not self.vector_store.pinecone_environment:
                         errors.append("PINECONE_ENVIRONMENT required when VECTOR_STORE_TYPE=pinecone")

                if self.message_history.store_type == "redis":
                     # Redis connection will be tested at runtime
                     pass

                return errors

# Global settings instance
settings = Settings()