

from ingestion.vectorstore import load_vectorstore, load_documents

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

VECTOR_K = 20

_db = load_vectorstore()

_vector_retriever = _db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": VECTOR_K,
        # "fetch_k": 40,
    },
)

# bm25 retriever

_documents = load_documents()

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

# cross encoder

_cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-large"
)

_reranker = CrossEncoderReranker(
    model=_cross_encoder,
    top_n=10,
)

# final retriever

_retriever = ContextualCompressionRetriever(
    base_compressor=_reranker,
    base_retriever=_hybrid_retriever,
)


def get_retriever_cross_encoder():
    return _retriever

