

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
from retrieval.retriever import get_retriever

def run_retrieval_eval(ground_truth_path):

    with open(ground_truth_path) as f:
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



