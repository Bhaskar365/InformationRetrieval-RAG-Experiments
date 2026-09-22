

import json
from collections import defaultdict

from .config import K_VALUES

from .metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


def load_dataset(path):

    questions = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            questions.append(
                json.loads(line)
            )

    return questions


def evaluate(dataset, retriever):

    all_results = []

    max_k = max(K_VALUES)

    for index, item in enumerate(
        dataset
    ):

        question = item["question"]

        retrieved = retriever(question, max_k)

        relevant = {
            chunk["chunk_id"]
            for chunk in item["relevant_chunks"]
        }

        relevance_map = {}

        for chunk in item["relevant_chunks"]:

            if chunk["relevance"] == "primary":

                relevance_map[
                    chunk["chunk_id"]
                ] = 2

            else:

                relevance_map[
                    chunk["chunk_id"]
                ] = 1

        result = {
            "question_id":
                item["question_id"],

            "retrieved":
                retrieved,

            "relevant":
                list(relevant),
        }

        for k in K_VALUES:

            result[
                f"recall@{k}"
            ] = recall_at_k(
                retrieved,
                relevant,
                k
            )

            result[
                f"ndcg@{k}"
            ] = ndcg_at_k(
                retrieved,
                relevance_map,
                k
            )

        result[
            "reciprocal_rank"
        ] = reciprocal_rank(
            retrieved,
            relevant
        )

        all_results.append(
            result
        )

        print(
            f"Evaluated "
            f"{index + 1}/{len(dataset)}"
        )

    return all_results


def aggregate(results):

    metrics = defaultdict(list)

    for result in results:

        for key, value in result.items():

            if (
                key.startswith("recall@")
                or key.startswith("ndcg@")
                or key == "reciprocal_rank"
            ):

                metrics[key].append(
                    value
                )

    summary = {}

    for key, values in metrics.items():

        summary[key] = (
            sum(values) / len(values)
        )

    return summary


