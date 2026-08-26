

from ingestion.vectorstore import load_documents, load_vectorstore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

VECTOR_K = 20

_db = load_vectorstore()

# vector store

_vector_retriever = _db.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': VECTOR_K
    }
)

_documents = load_documents()

# bm25

_bm25_retriever = BM25Retriever.from_documents(
    _documents,
    k=VECTOR_K
)


_hybrid_retriever = EnsembleRetriever(
    retrievers=[
        _bm25_retriever,
        _vector_retriever
    ],
    weights=[
        0.5,
        0.5
    ]
)

def get_retriever_EnsembleRetriever():
    return _hybrid_retriever




