
from collections import defaultdict

def rrf(result_lists, k=60):
    scores = defaultdict(float)

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            scores[doc_id] += 1 / (k + rank)

    ranked = sorted(scores.items(),
                    key=lambda x: x[1],
                    reverse=True)

    return ranked

faiss = [
    {"id": 101, "score": 0.92},
    {"id": 55, "score": 0.90},
    {"id": 80, "score": 0.87},
]

bm25 = [
    {"id": 80, "score": 17.2},
    {"id": 33, "score": 15.8},
    {"id": 101, "score": 14.7},
]

results = rrf([faiss, bm25])

for doc, score in results:
    print(doc, score)
