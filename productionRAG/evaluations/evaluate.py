
import json

from retrieval.rag import ask_full

from collections import defaultdict
import os
from dotenv import load_dotenv

os.getenv()

GROUND_TRUTH_FILE = os.getenv("DATA_JSONL_PATH\\ground_truth.json")

with open(f'{GROUND_TRUTH_FILE}') as f:
    ground_truth = json.load(f)


results = []

for item in ground_truth:
    result = ask_full(
        item['question'],
        session_id=f"eval-{item['question_id']}"
    )

    results.append({
        "question_id": item["question_id"],
        "question": item["question"],
        "reference_answer": item["answer"],
        "generated_answer": result["answer"],
        "retrieved_chunks": result["context"],
        "relevant_chunks": item["relevant_chunks"],
        "essential_chunks": item["essential_chunks"],
    })

