

# Runs:

# BM25
# Vector
# Hybrid
# BGE

# import json
# import math
# from collections import defaultdict
# import os

# from retrieval.rag import ask

# from dotenv import load_dotenv

# load_dotenv()

# os.getenv()

# GROUND_TRUTH_FILE = os.getenv("DATA_JSONL_PATH\\ground_truth.json")
# RAG_OUTPUT_FILE = "DATA_JSONL_PATH\\questionAnalysis.jsonl"
# K_VALUES = [1,3,5,10,20]

# def precision_at_k(retrieved, relevant, k):

#     retrieved_k = retrieved[:k]
#     relevant = set(relevant)

#     hits = sum(chunk in relevant for chunk in retrieved_k)

#     return hits / k


# def recall_at_k(retrieved, relevant, k):

#     retrieved_k = retrieved[:k]
#     relevant = set(relevant)

#     if not relevant:
#         return 0.0

#     hits = sum(chunk in relevant for chunk in retrieved_k)

#     return hits / len(relevant)

# def essential_recall_at_k(retrieved, essential, k):

#     retrieved_k = set(retrieved[:k])
#     essential = set(essential)

#     if not essential:
#         return 0.0

#     return len(retrieved_k & essential) / len(essential)


# def reciprocal_rank(retrieved, relevant, k):
#     relevant = set(relevant)

#     for rank, chunk in enumerate(retrieved[:k], start=1):
#         if chunk in relevant:
#             return 1.0 / rank

#     return 0.0


# import json

# SESSION_ID = "cli-session"

# with open(f'DATA_JSONL_PATH\\ground_truth.json') as f:
#     ground_truth = json.load(f)

# results = []

# for item in ground_truth:
#     question = item['question']

#     rag_result = rag_system(question)

#     retrieved = rag_result['retrieved_chunks']
#     generated_answer = rag_result['answer']

#     result = {
#         "question_id": item['question_id'],
#         "precision@5": precision_at_k(retrieved, item['relevant_chunks'], 5),
#         'recall@5': recall_at_k(retrieved, item['relevant_chunks'], 5),
#         'essential_recall@5': essential_recall_at_k(retrieved, item['essential_chunks'], 5),
#         "mrr@5": reciprocal_rank(retrieved, item['relevant_chunks'], 5),
#         "generated_answer": generated_answer
#     }

#     results.append(result)

# import numpy as np

# metrics = [
#     "precision@5",
#     "recall@5",
#     "essential_recall@5",
#     "mrr@5"
# ]

# for metric in metrics:
#     score = np.mean([r[metric] for r in results])
#     print(f"{metric}: {score:.4f}")



#     # for chunk in ask(question, session_id=SESSION_ID):

#     #     if chunk["type"] == "answer":

#     #         rag_result = {
#     #             chunk["content"],

#     #         }

            

#         # rag_result = 


import json
# from retrieval.retriever import get_retriever
from retrieval.retriever import get_retriever
import pandas as pd
import os

from evaluations.vectorOnlyEvaluations import get_retriverVectorOnly

from dotenv import load_dotenv

load_dotenv()

# GROUND_TRUTH_FILE = os.getenv("DATA_JSONL_PATH\ground_truth.json")

GROUND_TRUTH_FILE = "D:\\mlTesting\\FAISS\\productionRAG\\data\\ground_truth.json"
VECTOR_ONLY_FILE = "D:\\mlTesting\\FAISS\\productionRAG\\reports\\vectorstoreOnly_report.jsonl"

def run_retrieval_eval_vectorOnly(ground_truth_path):

    with open(ground_truth_path, encoding='utf-8') as f:
        data = json.load(f)

        vector_only_retriever = get_retriverVectorOnly()

        eval_results = []

        for item in data:
            question = item['question']

            relevant_ids = set(item['relevant_chunks'])

            docs = vector_only_retriever.invoke(question)
            retrieved_ids = [doc.metadata.get('chunk_id') for doc in docs]

            eval_results.append({
                "question_id": item['question_id'],
                "question": question,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids
            })

        return eval_results

def run_retrieval_eval(ground_truth_path):

    with open(ground_truth_path, encoding='utf-8') as f:
        gt_data = json.load(f)

        retriever = get_retriever()

        eval_results = []

        for item in gt_data:
            question = item['question']

            relevant_ids = set(item['relevant_chunks'])

            docs = retriever.invoke(question) 
            retrieved_ids = [doc.metadata.get('chunk_id') for doc in docs]

            eval_results.append({
                "question_id": item["question_id"],
                "question": question,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids
            })

        return eval_results


def precision_at_k(retrieved_ids, relevant_ids, k):

    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)

    return hits / k

def recall_at_k(retrieved_ids, relevant_ids, k):

    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)

    return hits / len(relevant_ids) if relevant_ids else 0.0

def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            return 1.0 / rank

    return 0.0

def hit_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return int(any(cid in retrieved_ids for cid in top_k))

# eval_results = run_retrieval_eval(f"{GROUND_TRUTH_FILE}")
eval_results = run_retrieval_eval_vectorOnly(f"{GROUND_TRUTH_FILE}")

scored = []

for r in eval_results:
    scored.append({
        "question_id": r['question_id'],
        "precision@3": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 3),
        "recall@3": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 3),
        "precision@5": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 5),
        "recall@5": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 5),
        "precision@7": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 7),
        "recall@7": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 7),
        "precision@10": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 10),
        "recall@10": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 10),
        "mrr@5": reciprocal_rank_at_k(r["retrieved_ids"], r["relevant_ids"],3),
        "mrr@5": reciprocal_rank_at_k(r["retrieved_ids"], r["relevant_ids"],5),
        "mrr@5": reciprocal_rank_at_k(r["retrieved_ids"], r["relevant_ids"],7),
        "mrr@5": reciprocal_rank_at_k(r["retrieved_ids"], r["relevant_ids"],10),
        "hit@1": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 1),
        "hit@3": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 3),
        "hit@5": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 5),
        "hit@10": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 10),
    })

df = pd.DataFrame(scored)
print(df)

with open("D:\\mlTesting\\FAISS\\productionRAG\\reports\\vectorstoreOnly_report.jsonl", "w", encoding='utf-8') as f:
    f.write(json.dumps(scored)+ "\n")

print("\nAverages:\n", df.mean(numeric_only=True))



# from ingestion.vectorstore import load_vectorstore

# db = load_vectorstore()
# for cid in ["attentionisallyouneed_c9", "attentionisallyouneed_c10", "attentionisallyouneed_c11"]:
#     result = db.get(where={"chunk_id": cid})
#     print(cid, result["documents"])