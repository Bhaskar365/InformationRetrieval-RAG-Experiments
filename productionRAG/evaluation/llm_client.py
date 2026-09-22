

#---------------------------
# Original
# from langchain_ollama import ChatOllama

# from dotenv import load_dotenv

# load_dotenv()

# from .config import (
#     GENERATION_MODEL,
#     CRITIC_MODEL,
#     EMBEDDING_MODEL,
# )

# from .schemas import (
#     EvaluationBatch,
#     Critique,
# )

# from sentence_transformers import SentenceTransformer

# class LLMClient:

#     def __init__(self):

#         self.client = ChatOllama(
#                 model=GENERATION_MODEL,
#                 temperature=0.0
#             )

#         self.structured_client = (
#             self.client.with_structured_output(
#                 EvaluationBatch
#             )
#         )

#         self.critic_client = ChatOllama(
#                 model=CRITIC_MODEL,
#                 temperature=0.0
#             )

#         self.structured_critic_client = (
#             self.critic_client.with_structured_output(
#                 Critique
#             )
#         )

#         self.embedding_client = SentenceTransformer(EMBEDDING_MODEL)
        
#     def generate_questions(
#         self,
#         system_prompt: str,
#         user_prompt: str,
#     ) -> EvaluationBatch:


#         response = self.structured_client.invoke(
#             [
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": user_prompt,
#                 },
#             ]
#         )

#         return response

#     def critique_question(
#         self,
#         system_prompt: str,
#         user_prompt: str,
#     ) -> Critique:

#         response = self.structured_critic_client.invoke(
#             [
#                 {
#                         "role": "system",
#                         "content": system_prompt,
#                 },
#                 {
#                         "role": "user",
#                         "content": user_prompt,
#                 },
#             ]
#         )
            
#         return response

#     def embed(
#         self,
#         texts: list[str],
#     ) -> list[list[float]]:

#         embeddings = self.embedding_client.encode(
#             texts,
#             convert_to_numpy=True
#         )

#         return embeddings.tolist()

"""
LLM Client with provider abstraction.
Supports Ollama (local) and OpenRouter (cloud) with unified interface.
"""

import os
import time
import asyncio
from typing import Optional, Type, Any, TypeVar, Callable, List
from contextlib import contextmanager
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from config.settings import settings, LLMConfig
from evaluation.schemas import EvaluationBatch, Critique

from .config import (
    GENERATION_MODEL,
    CRITIC_MODEL,
    EMBEDDING_MODEL,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_DELAY_SECONDS
)

from .schemas import EvaluationBatch, Critique
from .exceptions import LLMError, RateLimiter

load_dotenv()

T = TypeVar('T')

def retry_with_backoff(
        max_attempts:int = LLM_RETRY_ATTEMPTS,
        base_delay:float = LLM_RETRY_DELAY_SECONDS,
        exceptions:tuple = (Exception)
) -> Callable:
    """Decorator for retrying LLM calls with exponential backoff."""

    def decorator(func: Callable[...,T]) -> Callable[...,T]:

        @wraps
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = 0
                    if attempt < max_attempts-1:
                        delay = base_delay * (2**attempt)
                        time.sleep(delay)
                    continue
            raise LLMError(f"Failed after {max_attempts} attempts: {last_exception}")

        return wrapper
    
    return decorator    

class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_minute:int=30):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time: Optional[float] = None

    def acquire(self):
        """Wait if necessary to comply with rate limit."""

        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()

class LLMFactory:
    """Factory for creating LLM clients based on configuration."""
    
    @staticmethod
    def create(config: Optional[LLMConfig] = None) -> BaseChatModel:
        """Create LLM client based on settings."""
        config = config or settings.llm
        
        if config.provider == "openrouter":
            return LLMFactory._create_openrouter_client(config)
        else:
            return LLMFactory._create_ollama_client(config)
    
    @staticmethod
    def _create_openrouter_client(config: LLMConfig) -> ChatOpenAI:
        """Create OpenRouter client (OpenAI-compatible)."""
        if not config.api_key:
            raise ValueError("OpenRouter API key required")
        
        return ChatOpenAI(
            model=config.generation_model,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
            # default_headers={
            #     "HTTP-Referer": "https://your-app.com",
            #     "X-Title": "Production RAG",
            # }
        )
    
    @staticmethod
    def _create_ollama_client(config: LLMConfig) -> ChatOllama:
        """Create local Ollama client."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("langchain-ollama not installed")
        
        return ChatOllama(
            model=config.generation_model,
            temperature=config.temperature,
            base_url=config.base_url,
        )


class LLMClient:
    """
    Unified LLM client with structured output support.
    Demonstrates provider-agnostic design.
    """
    
    def __init__(self):
        self.config = settings.llm
        self.client = LLMFactory.create(self.config)
        self.critic_client = LLMFactory.create(LLMConfig(
            provider=self.config.provider,
            api_key=self.config.api_key,
            generation_model=self.config.critic_model,
            temperature=self.config.temperature,
            base_url=self.config.base_url
        ))
        
        # Structured output clients (if model supports it)
        self.structured_client = self._try_structured_output(self.client, EvaluationBatch)
        self.structured_critic_client = self._try_structured_output(self.critic_client, Critique)

        self.rate_limiter = RateLimiter()
        self._init_clients()

    def _init_clients(self):
        """Initialize LLM clients."""
        try:
            self.client = ChatOllama(
                model=GENERATION_MODEL,
                temperature=0.0,
                timeout=120
            )
            self.structured_client = self.client.with_structured_output(
                EvaluationBatch,
                method='json_mode'
            )

            self.critic_client = ChatOllama(
                model=CRITIC_MODEL,
                temperature=0.0,
                timeout=120
            )
            self.structured_critic_client = self.structured_critic_client(
                Critique,
                method='json_mode'
            )

            self.embedding_client = SentenceTransformer(EMBEDDING_MODEL)

        except Exception as e:
            raise LLMError(f"Failed to initialize LLM clients: {e}")


        @retry_with_backoff(
            max_attempts=LLM_RETRY_ATTEMPTS,
            exceptions=(Exception, OutputParserException)
        )

        def generate_questions(self, system_prompt:str, user_prompt:str) -> EvaluationBatch:
            """Generate questions with rate limiting and retry."""
            self.rate_limiter.acquire()

            try:
                response = self.structured_client.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])
                return response
            
            except OutputParserException as e:
                raise LLMError(f"Failed to parse LLM output: {e}")

            except Exception as e:
                raise LLMError(f"Question generation failed: {e}")

        @retry_with_backoff(
            max_attempts=LLM_RETRY_ATTEMPTS,
            exceptions=(Exception, OutputParserException)
        )

        def critique_question(self, system_prompt:str, user_prompt:str) -> Critique:
            """Critique question with rate limiting and retry."""
            self.rate_limiter.acquire()

            try:
                response = self.structured_critic_client.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])

                return response

            except OutputParserException as e:
                raise LLMError(f"Failed to parse critique output: {e}")
            except Exception as e:
                raise LLMError(f"Critique generation failed: {e}")

        def embed(self, texts: List[str]) -> List[List[float]]:

            """Generate embeddings for texts."""
            if not texts:
                return []

            try:
                embeddings = self.embedding_client.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
                return embeddings.tolist()
            
            except Exception as e:
                raise LLMError(f"Embedding generation failed: {e}")


    def _try_structured_output(self, client: BaseChatModel, schema: Type[Any]):
        """Attempt to create structured output client, fallback to raw."""
        try:
            # Note: This only works with models that support function calling
            return client.with_structured_output(schema)
        except Exception as e:
            print(f"⚠️  Structured output not available: {e}")
            print("   Falling back to manual JSON parsing")
            return client
    
    def generate_questions(self, system_prompt: str, user_prompt: str) -> Any:
        """Generate questions with structured output."""
        if hasattr(self.structured_client, 'invoke'):
            return self.structured_client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        else:
            # Manual parsing fallback
            return self._manual_parse(
                self.client, 
                system_prompt, 
                user_prompt, 
                EvaluationBatch
            )
    
    def critique_question(self, system_prompt: str, user_prompt: str) -> Any:
        """Critique questions with structured output."""
        if hasattr(self.structured_critic_client, 'invoke'):
            return self.structured_critic_client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        else:
            return self._manual_parse(
                self.critic_client,
                system_prompt,
                user_prompt,
                Critique
            )
    
    def _manual_parse(self, client, system_prompt: str, user_prompt: str, schema: Type[Any]):
        """Fallback manual JSON parsing for models without function calling."""
        import json
        
        enhanced_prompt = f"""{system_prompt}

CRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanations.
Schema: {schema.model_json_schema()}
"""
        
        response = client.invoke([
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": user_prompt},
        ])
        
        content = response.content
        
        # Extract JSON from markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        try:
            return schema(**json.loads(content.strip()))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nContent: {content}")


# Backward compatibility
def get_llm_client():
    """Factory function for backward compatibility."""
    return LLMClient()# Original
#---------------------------





# from langchain_openai import ChatOpenAI  # Changed from ChatOllama
# import os
# from dotenv import load_dotenv
# load_dotenv()

# from .config import (
#     GENERATION_MODEL,
#     CRITIC_MODEL,
#     EMBEDDING_MODEL
# )
# from .schemas import (
#     EvaluationBatch,
#     Critique,
# )

# class LLMClient:
#     def __init__(self):
#         api_key = os.getenv("OPENROUTER_API_KEY")
#         if not api_key:
#             raise ValueError("OPENROUTER_API_KEY is required")
        
#         self.client = ChatOpenAI(
#             model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
#             temperature=0.0,
#             api_key=api_key,
#             base_url="https://openrouter.ai/api/v1"
#         )
        
#         self.structured_client = self.client.with_structured_output(
#             EvaluationBatch
#         )
        
#         self.critic_client = ChatOpenAI(
#             model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
#             temperature=0.0,
#             api_key=api_key,
#             base_url="https://openrouter.ai/api/v1"
#         )
        
#         self.structured_critic_client = self.critic_client.with_structured_output(
#             Critique
#         )
        
#         from sentence_transformers import SentenceTransformer
#         self.embedding_client = SentenceTransformer(EMBEDDING_MODEL)
    
#     def generate_questions(
#         self,
#         system_prompt: str,
#         user_prompt: str,
#     ) -> EvaluationBatch:
#         response = self.structured_client.invoke(
#             [
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": user_prompt,
#                 },
#             ]
#         )
#         return response
    
#     def critique_question(
#         self,
#         system_prompt: str,
#         user_prompt: str,
#     ) -> Critique:
#         response = self.structured_critic_client.invoke(
#             [
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": user_prompt,
#                 },
#             ]
#         )
#         return response
    
#     def embed(
#         self,
#         texts: list[str],
#     ) -> list[list[float]]:
#         embeddings = self.embedding_client.encode(
#             texts,
#             convert_to_numpy=True
#         )
#         return embeddings.tolist()