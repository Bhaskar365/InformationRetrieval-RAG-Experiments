
from evaluations.dataset import load_eval_dataset
from productionRAG.app import modelInput

def run_evaluation(dataset_path):

    dataset = load_eval_dataset(dataset_path)

    results = []

    for example in dataset:

        result = modelInput(example['question'])

        evaluation = evaluate_result(
            question=example["question"],
            expected_answer=example["expected_answer"],
            actual_answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"]
        )

        results.append({
            "id": example["id"],
            "question": example["question"],
            "answer": result["answer"],
            "evaluation": evaluation
        })

    return results