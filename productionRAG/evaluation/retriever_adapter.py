

from evaluations.vectorOnlyEvaluations import (
    get_retriverVectorOnly
)

from evaluations.vector_with_bm25_evaluation import (
    get_vectorPlusBM25
)

from evaluations.vectorBM25WithEnsembleRRFEvaluation import (
    get_retriever_EnsembleRetriever
)

from evaluations.fullEvaluationWithCrossEncoder import (
    get_retriever_cross_encoder
)


class RetrieverAdapter:
    """
    Adapter around the existing RAG retrievers.
      
    Every method returns only ranked chunk IDs.
    """

    def __init__(self):

        print("Initializing retrievers...")

        self.vector_retriever = (
            get_retriverVectorOnly()
        )

        self.bm25_retriever = (
            get_vectorPlusBM25()
        )

        self.rrf_retriever = (
            get_retriever_EnsembleRetriever()
        )

        self.cross_encoder_retriever = (
            get_retriever_cross_encoder()
        )

        print("All retrievers initialized successfully.")



    @staticmethod
    def _extract_chunk_ids(docs, k):
        """
        Convert LangChain Documents into a ranked list
        of chunk IDs.

        Input:
            [
                Document(...),
                Document(...),
                ...
            ]

        Output:
            [
                "chunk_001",
                "chunk_042",
                "chunk_103"
            ]
        """

        chunk_ids = []

        for doc in docs[:k]:

            chunk_id = doc.metadata.get("chunk_id")

            if chunk_id is not None:
                chunk_ids.append(chunk_id)

        return chunk_ids


    def vector_only(
        self,
        query: str,
        k: int = 20
    ) -> list[str]:
        """
        Dense/vector-only retrieval.

        Returns:
            Ranked chunk IDs.
        """

        docs = self.vector_retriever.invoke(query)

        return self._extract_chunk_ids(
            docs,
            k
        )

    # =============================================================
    # BM25 ONLY
    # =============================================================

    def bm25_only(
        self,
        query: str,
        k: int = 20
    ) -> list[str]:
        """
        BM25-only retrieval.

        Returns:
            Ranked chunk IDs.
        """

        docs = self.bm25_retriever.invoke(query)

        return self._extract_chunk_ids(
            docs,
            k
        )

    # =============================================================
    # VECTOR + BM25 + RRF
    # =============================================================

    def rrf(
        self,
        query: str,
        k: int = 20
    ) -> list[str]:
        """
        Hybrid retrieval using:

            Dense retrieval
                    +
                BM25 retrieval
                    ↓
                   RRF

        RRF is useful because vector and BM25 scores
        do not need to be placed on the same numerical scale.

        Returns:
            Ranked chunk IDs.
        """

        docs = self.rrf_retriever.invoke(query)

        return self._extract_chunk_ids(
            docs,
            k
        )

    # =============================================================
    # VECTOR + BM25 + RRF + CROSS ENCODER
    # =============================================================

    def cross_encoder(
        self,
        query: str,
        k: int = 20
    ) -> list[str]:
        """
        Full retrieval pipeline:

            Dense
              +
            BM25
              ↓
             RRF
              ↓
        BGE Cross Encoder
              ↓
             Top K

        Returns:
            Ranked chunk IDs.
        """

        docs = self.cross_encoder_retriever.invoke(query)

        return self._extract_chunk_ids(
            docs,
            k
        )


# =================================================================
# OPTIONAL TEST
# =================================================================

if __name__ == "__main__":

    adapter = RetrieverAdapter()

    query = "What is the main contribution of this paper?"

    K = 20

    print("\n" + "=" * 80)
    print("VECTOR ONLY")
    print("=" * 80)

    vector_ids = adapter.vector_only(
        query,
        K
    )

    for rank, chunk_id in enumerate(
        vector_ids,
        start=1
    ):
        print(f"{rank:02d}. {chunk_id}")


    print("\n" + "=" * 80)
    print("BM25 ONLY")
    print("=" * 80)

    bm25_ids = adapter.bm25_only(
        query,
        K
    )

    for rank, chunk_id in enumerate(
        bm25_ids,
        start=1
    ):
        print(f"{rank:02d}. {chunk_id}")


    print("\n" + "=" * 80)
    print("RRF HYBRID")
    print("=" * 80)

    rrf_ids = adapter.rrf(
        query,
        K
    )

    for rank, chunk_id in enumerate(
        rrf_ids,
        start=1
    ):
        print(f"{rank:02d}. {chunk_id}")


    print("\n" + "=" * 80)
    print("CROSS ENCODER")
    print("=" * 80)

    cross_encoder_ids = adapter.cross_encoder(
        query,
        K
    )

    for rank, chunk_id in enumerate(
        cross_encoder_ids,
        start=1
    ):
        print(f"{rank:02d}. {chunk_id}")

