

from ingestion.vectorstore import load_vectorstore

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_retriever():

    db = load_vectorstore()

    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 40}
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=5
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    # MMR only used if duplicates prob is very high, otherwise good data may get discarded
    # return db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k" : 5,
    #         "fetch_k" : 20
    #     }
    # )

    