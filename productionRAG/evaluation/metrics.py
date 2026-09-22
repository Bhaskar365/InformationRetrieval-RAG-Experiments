
import math


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:

    if not relevant:
        return 0.0

    retrieved_k = set(
        retrieved[:k]
    )

    hits = (
        retrieved_k
        & relevant
    )

    return (
        len(hits) / len(relevant)
    )


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
) -> float:

    for rank, chunk_id in enumerate(
        retrieved,
        start=1
    ):

        if chunk_id in relevant:

            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:

    score = 0.0

    for index, chunk_id in enumerate(
        retrieved[:k]
    ):

        rel = relevance.get(
            chunk_id,
            0
        )

        score += (
            rel
            / math.log2(index + 2)
        )

    return score


def ndcg_at_k(
    retrieved: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:

    actual = dcg_at_k(
        retrieved,
        relevance,
        k
    )

    ideal_relevances = sorted(
        relevance.values(),
        reverse=True
    )

    ideal = 0.0

    for index, rel in enumerate(
        ideal_relevances[:k]
    ):

        ideal += (
            rel
            / math.log2(index + 2)
        )

    if ideal == 0:
        return 0.0

    # return actual / ideal

    result = actual / ideal

    if result > 1.0:

        print("\n========== NDCG BUG ==========")
        print("K:", k)
        print("Retrieved:", retrieved[:k])
        print("Relevance:", relevance)
        print("Actual DCG:", actual)
        print("Ideal relevances:", ideal_relevances[:k])
        print("Ideal DCG:", ideal)
        print("NDCG:", result)
        print("===============================\n")

    return result
