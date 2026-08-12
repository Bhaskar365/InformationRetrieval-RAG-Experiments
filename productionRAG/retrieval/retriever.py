

from ingestion.vectorstore import load_vectorstore, load_documents

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# OLDER CODE

# def get_retriever():

#     db = load_vectorstore()

#     base_retriever = db.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 20, "fetch_k": 40}
#     )

#     cross_encoder = HuggingFaceCrossEncoder(
#         model_name="BAAI/bge-reranker-large"
#     )

#     reranker = CrossEncoderReranker(
#         model=cross_encoder,
#         top_n=5
#     )

#     return ContextualCompressionRetriever(
#         base_compressor=reranker,
#         base_retriever=base_retriever
#     )

    # MMR only used if duplicates prob is very high, otherwise good data may get discarded
    # return db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k" : 5,
    #         "fetch_k" : 20
    #     }
    # )

# retriever.py

# vector retriever

_db = load_vectorstore()

_vector_retriever = _db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 20,
        "fetch_k": 40,
    },
)

# bm25 retriever

_documents = load_documents()

_bm25_retriever = BM25Retriever.from_documents(
    _documents,
    k=20
)

# Hybrid retriever

_hybrid_retriever = EnsembleRetriever(
    retrievers=[
        _bm25_retriever,
        _vector_retriever
    ],
    weights=[
        0.4,
        0.6
    ]
)

# cross encoder

_cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-large"
)

_reranker = CrossEncoderReranker(
    model=_cross_encoder,
    top_n=5,
)

# final retriever

_retriever = ContextualCompressionRetriever(
    base_compressor=_reranker,
    base_retriever=_hybrid_retriever,
)


def get_retriever():
    return _retriever

def debug_retrieval(query):

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)


    # =====================================================
    # STEP 1 — CHROMA
    # =====================================================

    print("\n")
    print("=" * 80)
    print("STEP 1 — CHROMA VECTOR SEARCH")
    print("=" * 80)

    vector_docs = _vector_retriever.invoke(query)

    print(
        f"\nChroma returned {len(vector_docs)} documents"
    )

    for i, doc in enumerate(vector_docs, start=1):

        print("\n" + "-" * 80)
        print(f"VECTOR RESULT {i}")
        print("-" * 80)

        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("document_id:", doc.metadata.get("document_id"))
        print("filename:", doc.metadata.get("filename"))
        print("page:", doc.metadata.get("page_number"))

        print("\nCONTENT:")
        print(doc.page_content[:500])


    # =====================================================
    # STEP 2 — BM25
    # =====================================================

    print("\n")
    print("=" * 80)
    print("STEP 2 — BM25 KEYWORD SEARCH")
    print("=" * 80)

    bm25_docs = _bm25_retriever.invoke(query)

    print(
        f"\nBM25 returned {len(bm25_docs)} documents"
    )

    for i, doc in enumerate(bm25_docs, start=1):

        print("\n" + "-" * 80)
        print(f"BM25 RESULT {i}")
        print("-" * 80)

        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("document_id:", doc.metadata.get("document_id"))
        print("filename:", doc.metadata.get("filename"))
        print("page:", doc.metadata.get("page_number"))

        print("\nCONTENT:")
        print(doc.page_content[:500])


    # =====================================================
    # STEP 3 — HYBRID
    # =====================================================

    print("\n")
    print("=" * 80)
    print("STEP 3 — HYBRID RETRIEVAL")
    print("=" * 80)

    hybrid_docs = _hybrid_retriever.invoke(query)

    print(
        f"\nHybrid returned {len(hybrid_docs)} documents"
    )

    for i, doc in enumerate(hybrid_docs, start=1):

        print("\n" + "-" * 80)
        print(f"HYBRID RESULT {i}")
        print("-" * 80)

        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("document_id:", doc.metadata.get("document_id"))
        print("filename:", doc.metadata.get("filename"))
        print("page:", doc.metadata.get("page_number"))

        print("\nCONTENT:")
        print(doc.page_content[:500])


    # =====================================================
    # STEP 4 — BGE RERANKER
    # =====================================================

    print("\n")
    print("=" * 80)
    print("STEP 4 — BGE RERANKER")
    print("=" * 80)

    final_docs = _retriever.invoke(query)

    print(
        f"\nBGE returned {len(final_docs)} documents"
    )

    for i, doc in enumerate(final_docs, start=1):

        print("\n" + "-" * 80)
        print(f"FINAL RESULT {i}")
        print("-" * 80)

        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("document_id:", doc.metadata.get("document_id"))
        print("filename:", doc.metadata.get("filename"))
        print("page:", doc.metadata.get("page_number"))

        print("\nCONTENT:")
        print(doc.page_content[:1000])

    print("\n")
    print("=" * 80)