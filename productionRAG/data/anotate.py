
import json
from retrieval.retriever import get_retriever
import pandas as pd
import os

from evaluations.vectorOnlyEvaluations import get_retriverVectorOnly
from evaluations.vector_with_bm25_evaluation import get_vectorPlusBM25

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

def run_retrieval_eval_vectorPlusBM25(ground_truth_path):

    with open(ground_truth_path, encoding='utf-8') as f:
        data = json.load(f)

    # vector_plus_bm25_data = get_vectorPlusBM25()
    vector_retriever, bm25_retriever = get_vectorPlusBM25()

    eval_results = []

    for item in data:
        question = item['question']

        relevant_ids = set(item['relevant_chunks'])

        # docs = vector_plus_bm25_data.invoke(question)
        vector_docs = vector_retriever.invoke(question)
        bm25_docs = bm25_retriever.invoke(question)

        vector_ids = [
            doc.metadata['chunk_id']
            for doc in vector_docs
        ]

        bm25_ids = [
            doc.metadata['chunk_id']
            for doc in bm25_docs
        ]

        vector_id_set = set(vector_ids)
        bm25_id_set = set(bm25_ids)

        overlap_ids = list(
            vector_id_set & bm25_id_set
        )

        union_ids = list(vector_ids)

        for cid in bm25_ids:
            if cid not in union_ids:
                union_ids.append(cid)

        vector_set = set(vector_ids)
        bm25_set = set(bm25_ids)
        relevant_set = set(relevant_ids)

        vector_relevant = vector_set & relevant_set
        bm25_relevant = bm25_set & relevant_set

        bm25_unique_relevant = (
            bm25_set - vector_set
        ) & relevant_set

        print(
            item["question_id"],
            "Vector relevant:", vector_relevant,
            "BM25 relevant:", bm25_relevant,
            "BM25 unique relevant:", bm25_unique_relevant
        )

        eval_results.append({
            "question_id": item["question_id"],
            "question": question,

            "vector_ids": vector_ids,
            "bm25_ids": bm25_ids,

            "overlap_ids": overlap_ids,
            "union_ids": union_ids,

            "relevant_ids": list(relevant_ids),

            "vector_relevant": [
                cid for cid in vector_ids
                if cid in relevant_ids
            ],

            "bm25_relevant": [
                cid for cid in bm25_ids
                if cid in relevant_ids
            ],

            "union_relevant": [
                cid for cid in union_ids
                if cid in relevant_ids
            ],

                # Diagnostic metrics
            "vector_relevant_count": len(
                set(vector_ids) & set(relevant_ids)
            ),

            "bm25_relevant_count": len(
                set(bm25_ids) & set(relevant_ids)
            ),

            "overlap_relevant_count": len(
                (set(vector_ids) & set(bm25_ids)) & set(relevant_ids)
            ),

            "bm25_unique_relevant_count": len(
                (set(bm25_ids) - set(vector_ids)) & set(relevant_ids)
            ),

            "vector_unique_relevant_count": len(
                (set(vector_ids) - set(bm25_ids)) & relevant_ids
            ),

            "retrieved_ids": union_ids
        })


        print("BM25 relevant count:",
            len(set(bm25_ids) & set(relevant_ids)))

        print("Overlap relevant count:",
            len(set(vector_ids) & set(bm25_ids) & set(relevant_ids)))

        print("BM25 unique relevant count:",
            len((set(bm25_ids) - set(vector_ids)) & set(relevant_ids)))

        print("Vector unique relevant count:",
            len((set(vector_ids) - set(bm25_ids)) & set(relevant_ids)))


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
    return int(any(cid in relevant_ids for cid in top_k))

# eval_results = run_retrieval_eval(f"{GROUND_TRUTH_FILE}")
# eval_results = run_retrieval_eval_vectorOnly(f"{GROUND_TRUTH_FILE}")
eval_results = run_retrieval_eval_vectorPlusBM25(f"{GROUND_TRUTH_FILE}")

scored = []

print("\n")
print("=" * 80)
print("VECTOR vs BM25 DIAGNOSTIC")
print("=" * 80)

for r in eval_results:

    # print(
    #     f"\n{r['question_id']}"
    #     f" | Vector relevant: {r['vector_relevant_count']}"
    #     f" | BM25 relevant: {r['bm25_relevant_count']}"
    #     f" | Overlap: {r['overlap_relevant_count']}"
    #     f" | BM25 unique: {r['bm25_unique_relevant_count']}"
    #     f" | Vector unique: {r['vector_unique_relevant_count']}"
    # )

    scored.append({
        "question_id": r['question_id'],
        # "precision@3": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 3),
        "recall@3": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 3),
        # "precision@5": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 5),
        "recall@5": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 5),
        # "precision@7": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 7),
        "recall@7": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 7),
        # "precision@10": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 10),
        "recall@10": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 10),
        # "precision@15": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 15),
        "recall@15": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 15),
        # "precision@20": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 20),
        "recall@20": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 20),

        # for vector+bm25
        "precision@3": precision_at_k(r['vector_ids'], r['relevant_ids'], 3),
        "precision@5": precision_at_k(r['vector_ids'], r['relevant_ids'], 5),
        "precision@7": precision_at_k(r['vector_ids'], r['relevant_ids'], 7),
        "precision@10": precision_at_k(r['vector_ids'], r['relevant_ids'], 10),
        "precision@15": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 15),
        "precision@20": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 20),


        "mrr@3": reciprocal_rank_at_k(
            r["retrieved_ids"], 
            r["relevant_ids"],
            3),
        "mrr@5": reciprocal_rank_at_k(
            r["retrieved_ids"], 
            r["relevant_ids"],
            5),
        "mrr@7": reciprocal_rank_at_k(
            r["retrieved_ids"], 
            r["relevant_ids"],
            7),
        "mrr@10": reciprocal_rank_at_k(
                    r["retrieved_ids"], 
                    r["relevant_ids"],
                    10),
         "mrr@15": reciprocal_rank_at_k(
                    r["retrieved_ids"], 
                    r["relevant_ids"],
                    15),
        "mrr@20": reciprocal_rank_at_k(
                    r["retrieved_ids"], 
                    r["relevant_ids"],
                    20),

        

        "hit@1": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 1),
        "hit@3": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 3),
        "hit@5": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 5),
        "hit@10": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 10),
        "hit@15": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 15),
        "hit@20": hit_at_k(r["retrieved_ids"], r["relevant_ids"], 20),
        
    })

df = pd.DataFrame(scored)
print(df)

# with open("D:\\mlTesting\\FAISS\\productionRAG\\reports\\vectorstoreOnly_report.jsonl", "w", encoding='utf-8') as f:
#     f.write(json.dumps(scored)+ "\n")

with open("D:\\mlTesting\\FAISS\\productionRAG\\reports\\vectorstorePlustBM25_report.jsonl", "w", encoding='utf-8') as f:
    f.write(json.dumps(scored)+ "\n")

print("\nAverages:\n", df.mean(numeric_only=True))

