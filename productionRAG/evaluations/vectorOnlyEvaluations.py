

from ingestion.vectorstore import load_vectorstore

VECTOR_K = 20

_db = load_vectorstore()


_vector_retriever = _db.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k':VECTOR_K
    }
)

def get_retriverVectorOnly():
    return _vector_retriever
