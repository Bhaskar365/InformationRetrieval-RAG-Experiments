

from ingestion.vectorstore import load_vectorstore, load_documents

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

import json

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

# Hybrid retriever

#final_score = 0.4 × BM25_contribution + 0.6 × vector_contribution

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
    top_n=10,
)

# final retriever

_retriever = ContextualCompressionRetriever(
    base_compressor=_reranker,
    base_retriever=_hybrid_retriever,
)


def get_retriever():
    return _retriever

def print_chunks_for_annotation(query):

    docs = _db.similarity_search(query, k=20)

    for rank, doc in enumerate(docs, 1):

        record = {
            "rank": rank,
            "chunk_id": doc.metadata.get('chunk_id'),
            "page_number": doc.metadata.get('page_number'),
            "content": doc.page_content
        }

        # print(f"\n{'=' * 80}")
        # print(f"RANK: {rank}")
        # print(f"CHUNK ID: {doc.metadata.get('chunk_id')}")
        # print(f"PAGE: {doc.metadata.get('page_number')}")
        # print(f"\n{doc.page_content[:1000]}")

        with open("D:\\mlTesting\\FAISS\\productionRAG\\data\\questionAnalysis.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

def debug_retrieval(query):

    vector_results = _db.similarity_search_with_score(
        query,
        k=VECTOR_K
    )

    vector_data = {}

    for rank, (doc, score) in enumerate(vector_results, start=1):
        chunk_id = doc.metadata.get('chunk_id')

        vector_data[chunk_id] = {
            "rank": rank,
            "score": float(score)
        }

    print("similarity_search_with_score: ", vector_results)

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

    vector_chunks = [
        doc.metadata.get('chunk_id') for doc in vector_docs
    ]

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

        # record = {
        #     "query": query,
        #     # "answer": doc.metadata.get("page_content"),
        #     "content": doc.page_content[:100],
        #     "chunk_id": doc.metadata.get("chunk_id"),
        #     "document_id": doc.metadata.get("document_id"),
        #     "filename": doc.metadata.get("filename"),
        #     "page": doc.metadata.get("page_number"),
        #     # "citation_id": doc.metadata.get("citation_id"),
        #     # "content": doc.page_content,
        #     # "rank": i,
        # }

        # with open("D:\\mlTesting\\FAISS\\productionRAG\\data\\analysis.jsonl", "a", encoding='utf-8') as f:
        #     f.write(json.dumps(record) + "\n")

        # with open("D:\\mlTesting\\FAISS\\productionRAG\\data\\analysis.txt", "a", encoding='utf-8') as f:
        #     f.write(f"question:  {query} \n")
        #     f.write(f"Answer:  {doc.page_content} \n")
        #     f.write(f"metadata: {doc.metadata} \n")
        #     f.write(f"citation_id:  {doc.metadata['chunk_id']} \n")
        #     f.write(f"document_id:  {doc.metadata['document_id']} \n")
        #     f.write(f"page:  {doc.metadata['page_number']} \n")
        #     f.write(f"filename:  {doc.metadata['filename']} \n")

        #     f.write(str(print("--------------------------------------------------")))
        #     f.write(str(print("",end='\n\n')))


    # =====================================================
    # STEP 2 — BM25
    # =====================================================

    print("\n")
    print("=" * 80)
    print("STEP 2 — BM25 KEYWORD SEARCH")
    print("=" * 80)

    bm25_docs = _bm25_retriever.invoke(query)

    bm25_chunks = [
        doc.metadata.get('chunk_id') for doc in bm25_docs
    ]

    print(
        f"\nBM25 returned {len(bm25_docs)} documents"
    )

    bm25_data = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        chunk_id = doc.metadata.get('chunk_id')

        bm25_data[chunk_id] = {
            "rank" : rank
        }


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

    hybrid_chunks = [
        doc.metadata.get('chunk_id') for doc in hybrid_docs
    ]

    hybrid_data = {}

    for rank, doc in enumerate(hybrid_docs, start=1):
        chunk_id = doc.metadata.get('chunk_id')

        hybrid_data[chunk_id] = {
            "rank": rank
        }

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

    bge_chunks = [
        doc.metadata.get('chunk_id') for doc in final_docs
    ]

    bge_data = {}

    for rank, doc in enumerate(final_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")

        bge_data[chunk_id] = {
            "rank": rank
        }

    print(
        f"\nBGE returned {final_docs}"
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

        
        record = {
            "query": query,
            # "answer": doc.metadata.get("page_content"),
            "content": doc.page_content[:100],
            "chunk_id": doc.metadata.get("chunk_id"),
            "document_id": doc.metadata.get("document_id"),
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page_number"),
            # "citation_id": doc.metadata.get("citation_id"),
            # "content": doc.page_content,
            "rank": i,

            "vector_chunks": vector_chunks,

            "bm25_chunks": bm25_chunks,

            "hybrid_chunks": hybrid_chunks,

            "bge_chunks": bge_chunks,

            "vector_rank": vector_data.get(chunk_id, {}).get('rank'),
            "vector_score": vector_data.get(chunk_id, {}).get("score"),

            "bm25_rank": bm25_data.get(chunk_id, {}).get('rank'),

            "hybrid_rank": hybrid_data.get(chunk_id, {}).get("rank"),

            "bge_rank": bge_data.get(chunk_id, {}).get("rank"),

            "relevance":False,
        }

        with open("D:\\mlTesting\\FAISS\\productionRAG\\data\\analysis.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

    # print("\n")
    # print("=" * 80)