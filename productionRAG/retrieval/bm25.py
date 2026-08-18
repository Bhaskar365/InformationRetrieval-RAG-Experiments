
from langchain_community.retrievers import BM25Retriever

def create_bm25_retriever(documents,  k=20):

    retriever = BM25Retriever.from_documents(
        documents,
        k=k
    )

    return retriever


def get_bm25_scores(documents):
    scores = BM25Retriever.get_scores(documents)

    results = []

    for idx, score in enumerate(scores):
        if score > 0:
            results.append({
                "chunk": documents[idx],
                "bm25_score": float(score)
            })
