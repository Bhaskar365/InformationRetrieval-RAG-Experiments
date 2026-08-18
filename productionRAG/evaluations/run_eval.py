

from evaluations.evaluator import run_evaluation
import json


results = run_evaluation(
    "data/eval_dataset.json"
)


with open(
    "results/raw/evaluation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Evaluation complete.")

