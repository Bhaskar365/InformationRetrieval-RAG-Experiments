
import json
from retrieval.retriever import get_retriever
import pandas as pd
import os
import math

from evaluations.vectorOnlyEvaluations import get_retriverVectorOnly
from evaluations.vector_with_bm25_evaluation import get_vectorPlusBM25
from evaluations.vectorBM25WithEnsembleRRFEvaluation import get_retriever_EnsembleRetriever

from dotenv import load_dotenv

load_dotenv()

# GROUND_TRUTH_FILE = os.getenv("DATA_JSONL_PATH\ground_truth.json")
GROUND_TRUTH_FILE = "D:\\mlTesting\\FAISS\\productionRAG\\data\\ground_truth.json"

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

def run_retrieval_eval_BM25(ground_truth_path):

    with open(ground_truth_path, encoding='utf-8') as f:
        data = json.load(f)

    bm25_retriever = get_vectorPlusBM25()

    eval_results = []

    for item in data:
        question = item['question']

        relevant_ids = set(item['relevant_chunks'])

        bm25_docs = bm25_retriever.invoke(question)

        bm25_ids = [
            doc.metadata['chunk_id']
            for doc in bm25_docs
        ]

        ks = [3, 5, 7, 10, 15, 20]

        ndcg_scores = {}

        for k in ks:
            ndcg_scores[f"ndcg@{k}"] = ndcg_at_k(
                bm25_ids,
                relevant_ids,
                k
            )

        retrieved_ids = bm25_ids    

        eval_results.append({
            "question_id": item["question_id"],
            "question": question,

            "bm25_ids": bm25_ids,

            "relevant_ids": list(relevant_ids),

            "bm25_relevant": [
                cid for cid in bm25_ids
                if cid in relevant_ids
            ],

            "bm25_relevant_count": len(
                set(bm25_ids) & set(relevant_ids)
            ),

            "ndcg_scores": {**ndcg_scores},

            "retrieved_ids": retrieved_ids
        })


        print("BM25 relevant count:",
            len(set(bm25_ids) & set(relevant_ids)))

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


def run_retrieval_eval_vectorPlusBM25PlusRRF(ground_truth_path):

    with open(ground_truth_path, encoding='utf-8') as f:
            data = json.load(f)

    retriever = get_retriever_EnsembleRetriever()

    eval_results = []

    for item in data:
        question = item['question']

        relevant_ids = set(item['relevant_chunks'])

        docs = retriever.invoke(question)
        retrieved_ids = [doc.metadata.get('chunk_id') for doc in docs]

        ks = [3, 5, 7, 10, 15, 20]
        
        ndcg_scores = {}
        
        for k in ks:
            ndcg_scores[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )

        assert len(retrieved_ids) == len(set(retrieved_ids)), (
                f"Duplicate chunk IDs for {item['question_id']}: {retrieved_ids}"
            )

        eval_results.append({
            "question_id": item['question_id'],
            "question": question,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "ndcg_scores": {**ndcg_scores},
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


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    Calculate binary nDCG@K.

    retrieved_ids: ranked list of retrieved chunk IDs
    relevant_ids: set/list of ground-truth relevant chunk IDs
    """

    relevant_ids = set(relevant_ids)

    top_k = retrieved_ids[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant_ids:
            relevance = 1
        else:
            relevance = 0

        dcg += relevance / math.log2(rank+1)

    ideal_relevant_count = min(len(relevant_ids), k)

    idcg = 0.0

    for rank in range(1, ideal_relevant_count+1):
        idcg += 1 / math.log2(rank+1)

    if idcg == 0:
        return 0.0

    return dcg/idcg    

# eval_results = run_retrieval_eval(f"{GROUND_TRUTH_FILE}")
# eval_results = run_retrieval_eval_vectorOnly(f"{GROUND_TRUTH_FILE}")
# eval_results = run_retrieval_eval_BM25(f"{GROUND_TRUTH_FILE}")
eval_results = run_retrieval_eval_vectorPlusBM25PlusRRF(f"{GROUND_TRUTH_FILE}")




scored = []

print("\n")
print("=" * 80)
print("VECTOR vs BM25 DIAGNOSTIC")
print("=" * 80)

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
        "precision@15": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 15),
        "recall@15": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 15),
        "precision@20": precision_at_k(r['retrieved_ids'], r['relevant_ids'], 20),
        "recall@20": recall_at_k(r["retrieved_ids"], r["relevant_ids"], 20),

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

        "ndcg@3": r['ndcg_scores']['ndcg@3'],
        "ndcg@5": r['ndcg_scores']['ndcg@5'],
        "ndcg@7": r['ndcg_scores']['ndcg@7'],
        "ndcg@10": r['ndcg_scores']['ndcg@10'],
        "ndcg@15": r['ndcg_scores']['ndcg@15'],
        "ndcg@20": r['ndcg_scores']['ndcg@20'],
        
    })

df = pd.DataFrame(scored)
print(df)

# with open("D:\\mlTesting\\FAISS\\productionRAG\\reports\\vectorOnly_report.jsonl", "w", encoding='utf-8') as f:
#     f.write(json.dumps(scored)+ "\n")

# with open("D:\\mlTesting\\FAISS\\productionRAG\\reports\\BM25_report.jsonl", "w", encoding='utf-8') as f:
#     f.write(json.dumps(scored)+ "\n")

with open("D:\\mlTesting\\FAISS\\productionRAG\\reports\\vector+BM25+RRF_report.jsonl", "w", encoding='utf-8') as f:
    f.write(json.dumps(scored)+ "\n")


print("\nAverages:\n", df.mean(numeric_only=True))

