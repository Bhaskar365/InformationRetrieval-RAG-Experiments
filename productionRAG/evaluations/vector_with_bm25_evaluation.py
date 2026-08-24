
from ingestion.vectorstore import load_vectorstore, load_documents
from langchain_community.retrievers import BM25Retriever

VECTOR_K = 20

_db = load_vectorstore()

# VECTOR STORE
_vector_retriever = _db.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': VECTOR_K
    }
)

# BM25
_documents = load_documents()

_bm25_documents = BM25Retriever.from_documents(
    _documents,
    k=VECTOR_K
)

# vector_ids = {
#    doc.metadata['doc_id'] for doc in _vector_retriever
# } 

# bm25_ids = {
#     doc.metadata['doc_id'] for doc in _bm25_documents
# }

# _overlap = vector_ids & bm25_ids


def get_vectorPlusBM25():
    return _vector_retriever, _bm25_documents