

import re


def normalize_question(
    question: str
) -> str:

    question = question.lower()

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    question = re.sub(
        r"[^\w\s]",
        "",
        question
    )

    return question.strip()


def exact_deduplicate(
    questions: list,
) -> list:

    seen = set()

    result = []

    for question in questions:

        normalized = normalize_question(
            question.question
        )

        if normalized in seen:
            continue

        seen.add(normalized)

        result.append(question)

    return result

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def semantic_deduplicate(
    questions: list,
    embeddings: list[list[float]],
    threshold: float = 0.90,
) -> list:

    if not questions:
        return []

    embeddings = np.array(
        embeddings
    )

    kept_indices = []

    for i in range(
        len(questions)
    ):

        if not kept_indices:

            kept_indices.append(i)

            continue

        current = embeddings[i:i + 1]

        previous = embeddings[
            kept_indices
        ]

        similarities = cosine_similarity(
            current,
            previous
        )[0]

        if max(similarities) < threshold:

            kept_indices.append(i)

    return [
        questions[i]
        for i in kept_indices
    ]
