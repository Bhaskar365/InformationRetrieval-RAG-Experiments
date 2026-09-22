

"""
Vector Store Factory Pattern Implementation.
Supports seamless switching between local (FAISS) and cloud (Pinecone) vector stores.
Demonstrates production-grade abstraction and dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import os
import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Pinecone imports (production)
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None
    PineconeVectorStore = None

from config.settings import settings

class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add_texts(self, text: List[str], metadatas: Optional[str[dict]]=None) -> List[str]:
         """Add texts to the vector store."""
         pass

    @abstractmethod
    def similarity_search(self, query:str, k:int=4)->List[Tuple[str,float]]:
        """Search for similar texts."""
        pass

    @abstractmethod
    def save(self, path:Optional[str]=None)->None:
        """Persist the vector store."""
        pass

    @abstractmethod
    def load(self, path:Optional[str]=None)->"BaseVectorStore":
        """Load a persisted vector store."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    Local FAISS vector store implementation.
    Ideal for development, testing, and small-scale deployments.
    Zero cost, zero external dependencies.
    """

    def __init__(self, embeddings:Embeddings, index_path:Optional[str]=None):
        self.embeddings = embeddings
        self.index_path = index_path or settings.vector_store.faiss_index_path
        self._store: Optional[FAISS] = None

    def _get_store(self) -> FAISS:
        if self._store is None:
            # Initialize empty store or load existing
            if os.path.exists(self.index_path):
                self._store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create empty store with a dummy document
                self._store = FAISS.from_texts(
                    ["initialization"],
                    self.embeddings
                )

        return self._store

    def add_texts(self, texts:List[str], metadatas: Optional[List[dict]]=None)->List[str]:
        """Add texts to FAISS index."""
        store = self._get_store()
        return store.add_texts(texts,)

    def similarity_search(self, query:str, k:int=4)->List[Tuple[str,float]]:
        """Search FAISS index with scores."""
        store = self._get_store()
        docs_with_score = store.similarity_search_with_score(query,k=k)
        return [(doc.page_content, score) for doc, score in docs_with_score]

    def save(self, path: Optional[str]=None)->None:
        """Save FAISS index to disk."""
        save_path = path or self.index_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._get_store().save_local(save_path)
        print(f"FAISS index saved to {save_path}")

    def load(self, path: Optional[str]=None)->"FAISSVectorStore":
        """Load FAISS index from disk."""
        load_path = path or self.index_path
        if os.path.exists(load_path):
            self._store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        print(f"FAISS index loaded from {load_path}")
        return self

    def as_retriever(self, search_kwargs: Optional[dict]=None):
        """Return LangChain retriever interface."""
        return self._get_store().as_retriever(search_kwargs=search_kwargs)

class PineconeVectorStoreWrapper(BaseVectorStore):

    """
    Production Pinecone vector store implementation.
    Demonstrates cloud-native vector database integration.
    """

    def __init__(self, embeddings:Embeddings):
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone packages not installed"
                "Run: pip install pinecone-client langchain-pinecone"
            )

        self.embeddings = embeddings
        self.config = settings.vector_store
        self._pc: Optional[Pinecone] = None
        self._index = None
        self._store: Optional[PineconeVectorStore] = None

        self._initialize()

    def _initialize(self):
        """Initialize Pinecone connection and ensure index exists."""
        print("Initializing Pinecone Vector Store...")

        self._pc = Pinecone(
            api_key=self.config.pinecone_api_key,
            environment=self.config.pinecone_environment
        )

        # Check if index exists
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]

        if self.config.pinecone_index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.config.pinecone_index_name}")
            self._pc.create_index(
                name=self.config.pinecone_index_name,
                dimension=self.config.pinecone_dimension,
                metric=self.config.pinecone_metric,
                spec=ServerlessSpec(
                    cloud=self.config.pinecone_cloud,
                    region=self.config.pinecone_region
                )
            )

            # wait for index to be ready
            import time

            while not self._pc.describe_index(self.config.pinecone_index_name).status['ready']:
                time.sleep(1)
                print("Waiting for index to be ready...")

        # Connect to index
        self._index = self._pc.index(self.config.pinecone_index_name)

        # Langchain wrapper
        self._store = PineconeVectorStore(
            self=self._index,
            embeddings=self.embeddings
        )

        print(f"Connected to Pinecone index: {self.config.pinecone_index_name}")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]]=None) -> List[str]:
        """Add texts to Pinecone."""
        return self._store.add_texts(texts, metadatas=metadatas)


    def similarity_search(self, query:str, k:int=4) -> List[Tuple[str, float]]:
        """Search Pinecone with scores."""
        docs_with_scores = self._store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in docs_with_scores]


    def save(self, path: Optional[str]=None) -> None:
        """No-op for Pinecone (cloud-persisted)."""
        print("Pinecone is cloud-persisted, no local save needed")

    def load(self, path: Optional[str]=None) -> "PineconeVectorStoreWrapper":
        """Re-initialize connection (Pinecone is always loaded from cloud)."""
        self._initialize()
        return self

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Return LangChain retriever interface."""
        return self._store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_all(self):
        """Delete all vectors from index (use with caution)."""
        self._index.delete(delete_all=True)
        print("All vectors deleted from Pinecone index")


class VectorStoreFactory:
    """
    Factory class for creating vector store instances.
    Demonstrates the Factory Pattern for dependency injection.
    """

    _stores:dict = {}    # singleton cache pattern

    @classmethod
    def create(cls, store_type: Optional[str]=None, embeddings: Optional[Embeddings]=None) -> BaseVectorStore:
        """
    Factory class for creating vector store instances.
    Demonstrates the Factory Pattern for dependency injection.
    """
    
    _stores: dict = {}  # Cache for singleton pattern
    
    @classmethod
    def create(cls, store_type: Optional[str] = None, embeddings: Optional[Embeddings] = None) -> BaseVectorStore:
        """
        Create a vector store instance based on configuration.
        
        Args:
            store_type: "faiss" or "pinecone" (defaults to settings)
            embeddings: Embedding model instance (creates default if None)
        
        Returns:
            BaseVectorStore 
        """

        store_type = store_type or settings.vector_store.store_type

        # create embeddings if not present
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding.model_name,
                model_kwargs={"device": settings.embedding.device}
            )


            # Check cache
            cache_key = f"{store_type}_{id(embeddings)}"
            if cache_key in cls._stores:
                return cls._stores[cache_key]

            # Create new instance
            if store_type == "pinecone":
                try:
                    store = PineconeVectorStoreWrapper(embeddings)
                    cls._stores[cache_key] = store
                    return store
                except Exception as e:
                    print(f"❌ Pinecone initialization failed: {e}")
                    print("⚠️  Falling back to FAISS...")
                    store_type = "faiss"

            # Default to FAISS
            store = FAISSVectorStore(embeddings)
            cls._stores[cache_key] = store
            return store

    @classmethod
    def get_embeddings(cls)->Embeddings:
        """Get default embedding model."""
        return HuggingFaceEmbeddings(
            model_name=settings.embedding.model_name,
            model_kwargs={"device": settings.embedding.device}
        )

    @classmethod
    def clear_cache(cls):
        """Clear the store cache (useful for testing)."""
        cls._stores.clear()

    # direct import convenience function
    def get_vector_store(emebddings: Optional[Embeddings]=None) -> BaseVectorStore:
        """Get configured vector store instance."""
        return VectorStoreFactory.create(embeddings=emebddings)

    