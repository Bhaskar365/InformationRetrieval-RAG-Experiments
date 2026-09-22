

import json

from evaluation.config import (
    FINAL_DIR,
    RESULTS_DIR,
)

from evaluation.run_evaluation import (
    load_dataset,
    evaluate,
    aggregate,
)

from evaluation.retriever_adapter import (
    RetrieverAdapter,
)


def save_json(data, path):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():

    dataset = load_dataset(FINAL_DIR / "eval_test.jsonl")

    adapter = RetrieverAdapter()

    retrievers = {
        
        "dense":
            adapter.vector_only,

        "bm25":
            adapter.bm25_only,

        "hybrid":
            adapter.rrf,

        "hybrid_reranked":
            adapter.cross_encoder,
    }

    comparison = {}

    for name, retriever in retrievers.items():

        print(
            f"\n========== {name} =========="
        )

        results = evaluate(dataset, retriever)

        summary = aggregate(results)

        comparison[name] = summary

        save_json(results, RESULTS_DIR / f"{name}_results.json")

        print(
            json.dumps(summary, indent=2)
        )

    save_json(comparison, RESULTS_DIR / "comparison.json")


if __name__ == "__main__":
    main()
