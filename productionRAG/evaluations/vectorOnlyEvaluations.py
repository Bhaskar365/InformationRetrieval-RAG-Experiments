

from ingestion.vectorstore import load_vectorstore, load_documents

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import EnsembleRetriever

import json

VECTOR_K = 20

_db = load_vectorstore()


_vector_retriever = _db.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k':VECTOR_K
    }
)

# _hybrid_retriever = EnsembleRetriever(
#     retrievers=[
#         _vector_retriever
#     ]
# )

# _retriver = ContextualCompressionRetriever(
#     base_retriever=_hybrid_retriever
# )

def get_retriverVectorOnly():
    return _vector_retriever
