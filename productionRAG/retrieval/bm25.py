
from langchain_community.retrievers import BM25Retriever

def create_bm25_retriever(documents,  k=20):

    retriever = BM25Retriever.from_documents(
        documents,
        k=k
    )

    return retriever

