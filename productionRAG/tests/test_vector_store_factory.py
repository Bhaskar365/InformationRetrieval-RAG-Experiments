
import pytest
from core.vector_store_factory import VectorStoreFactory, FAISSVectorStore

def test_faiss_factory_creation():
    """Test that the factory creates a local FAISS store by default."""
    store = VectorStoreFactory.create(store_type="faiss")
    assert isinstance(store, FAISSVectorStore)

def test_pinecone_fallback():
    """Test that Pinecone fallback works if credentials are missing."""
    store = VectorStoreFactory.create(store_type="pinecone")
    assert isinstance(store, FAISSVectorStore)
