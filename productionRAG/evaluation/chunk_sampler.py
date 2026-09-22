
import random
from collections import defaultdict


def group_by_paper(chunks: list[dict]) -> dict[str, list[dict]]:

    groups = defaultdict(list)

    for chunk in chunks:
        groups[chunk["document_id"]].append(chunk)

    return dict(groups)


def sample_single_chunks(chunks: list[dict], n:int) -> list[dict]:

    if n >= len(chunks):
        return chunks.copy()

    return random.sample(
        chunks,
        n
    )

def sample_multi_chunk_groups(chunks: list[dict], n: int, group_size: int = 2) -> list[list[dict]]:

    by_paper = group_by_paper(chunks)

    eligible = [
        paper_chunks
        for paper_chunks in by_paper.values()
        if len(paper_chunks) >= group_size
    ]

    groups = []

    if not eligible:
        return groups

    for _ in range(n):

        paper_chunks = random.choice(
            eligible
        )

        # Pick nearby chunks when possible.
        start = random.randint(
            0,
            len(paper_chunks) - group_size
        )

        group = paper_chunks[
            start:start + group_size
        ]

        groups.append(group)

    return groups

