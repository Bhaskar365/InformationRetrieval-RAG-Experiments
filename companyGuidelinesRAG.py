"""
Company Guidelines RAG pipeline
Hybrid retrieval (BM25 + dense FAISS) fused with Reciprocal Rank Fusion (RRF),
answered by a local Qwen2.5-0.5B-Instruct model.

Expected file format for companyGuidelines.txt (valid JSON):
[
  {"id": 1, "text": "Long paragraph of guideline text..."},
  {"id": 2, "text": "Another long paragraph..."}
]

Long text per id is split into overlapping chunks at ingestion time. Each
chunk keeps a reference back to its parent document id, so retrieval metrics
and citations can be reported at the document level even though search
happens at the chunk level.
"""

import json
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FILE_PATH = "companyGuidelines.txt"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # one embedder, used everywhere
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K_DENSE = 5
TOP_K_SPARSE = 5
TOP_K_FINAL = 5

CHUNK_SIZE_WORDS = 120   # words per chunk
CHUNK_OVERLAP_WORDS = 30 # overlap between consecutive chunks, preserves context across chunk boundaries

# ---------------------------------------------------------------------------
# Load + chunk data
# ---------------------------------------------------------------------------
with open(FILE_PATH, "r", encoding="utf-8") as f:
    guidelines = json.load(f)

raw_docs = [d for d in guidelines if d["id"] > 0]


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Word-based sliding-window chunking. Simple and dependency-free;
    swap for a sentence-aware splitter (e.g. nltk/spacy) if you need
    chunks to respect sentence boundaries."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


# chunk_id -> chunk text (what gets embedded / searched / shown as context)
document_store = {}
# chunk_id -> parent document id (for metrics + citing the source guideline)
chunk_to_parent = {}

for doc in raw_docs:
    doc_chunks = chunk_text(doc["text"])
    for i, chunk in enumerate(doc_chunks):
        chunk_id = f"{doc['id']}_{i}"
        document_store[chunk_id] = chunk
        chunk_to_parent[chunk_id] = doc["id"]

ids = list(document_store.keys())
texts = list(document_store.values())

# ---------------------------------------------------------------------------
# Sparse retrieval (BM25)
# ---------------------------------------------------------------------------
tokenized_corpus = [t.split(" ") for t in texts]
bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query, top_k=TOP_K_SPARSE):
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [{"id": ids[i], "score": float(scores[i])} for i in ranked_idx]


# ---------------------------------------------------------------------------
# Dense retrieval (FAISS)
# ---------------------------------------------------------------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME)

doc_embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(doc_embeddings)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product on normalized vecs = cosine sim
index.add(doc_embeddings)


def faiss_search(query, top_k=TOP_K_DENSE):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)
    return [
        {"id": ids[i], "score": float(s)}
        for s, i in zip(scores[0], idxs[0])
        if i != -1
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def rrf(result_lists, k=60):
    scores = defaultdict(float)
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result["id"]] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query, top_k=TOP_K_FINAL):
    dense_results = faiss_search(query)
    sparse_results = bm25_search(query)
    merged = rrf([dense_results, sparse_results])
    return merged[:top_k]  # list of (chunk_id, rrf_score)


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)


def generate_answer(query, contexts):
    context_block = "\n".join(contexts)
    prompt = f"""You are a company rules and guidelines expert.
If the question asked is covered by the guidelines below, provide the relevant guideline.
If it is not covered, respond exactly with "No such rule found."

Context:
{context_block}

Question:
{query}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=100, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def answer_query(query, top_k=TOP_K_FINAL):
    top_docs = retrieve(query, top_k=top_k)
    contexts = [document_store[chunk_id] for chunk_id, _ in top_docs]
    return generate_answer(query, contexts)


# ---------------------------------------------------------------------------
# Evaluation / metrics
# ---------------------------------------------------------------------------
# "Accuracy" isn't quite the right word for a ranking system — what you
# actually want is: did the retriever surface a relevant chunk near the top?
# The standard metrics for that are Hit Rate@k, Recall@k, Precision@k, and
# MRR. To compute any of them you need a small labeled eval set: for each
# test query, which parent document id(s) actually answer it.
#
# eval_set format:
# [
#   {"query": "How to reset the password", "relevant_ids": [3]},
#   {"query": "What is the leave policy",  "relevant_ids": [7, 8]},
#   ...
# ]
# relevant_ids refer to the *original* document ids (not chunk ids), since
# ground truth is usually labeled at the document level.

def evaluate_retrieval(eval_set, top_k=TOP_K_FINAL):
    hit_count = 0
    reciprocal_ranks = []
    precisions = []
    recalls = []

    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        retrieved = retrieve(query, top_k=top_k)
        retrieved_parent_ids = [chunk_to_parent[chunk_id] for chunk_id, _ in retrieved]

        # Hit rate: did *any* relevant doc show up in top_k?
        hit = any(pid in relevant for pid in retrieved_parent_ids)
        hit_count += int(hit)

        # MRR: reciprocal rank of the first relevant doc
        rr = 0.0
        for rank, pid in enumerate(retrieved_parent_ids, start=1):
            if pid in relevant:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

        # Precision@k / Recall@k (dedupe retrieved parent ids first,
        # since multiple chunks can come from the same document)
        unique_retrieved = set(retrieved_parent_ids)
        num_correct = len(unique_retrieved & relevant)
        precisions.append(num_correct / max(len(unique_retrieved), 1))
        recalls.append(num_correct / max(len(relevant), 1))

    n = len(eval_set)
    return {
        "hit_rate@k": hit_count / n,
        "mrr": sum(reciprocal_ranks) / n,
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
    }


def evaluate_generation(eval_set):
    """Rough proxy for generation quality: does the answer contain the
    expected keyword(s)/phrase? This is NOT a substitute for human review
    or an LLM-as-judge score, but it's a cheap smoke test you can run on
    every change. eval_set items here need an "expected_keywords" list."""
    correct = 0
    for item in eval_set:
        answer = answer_query(item["query"]).lower()
        if any(kw.lower() in answer for kw in item.get("expected_keywords", [])):
            correct += 1
    return correct / len(eval_set)


if __name__ == "__main__":
    query = "How to reset the password"
    print(answer_query(query))

    # Example evaluation run — replace with your own labeled queries
    eval_set = [
        {"query": "How to reset the password", "relevant_ids": [3]},
    ]
    print(evaluate_retrieval(eval_set))