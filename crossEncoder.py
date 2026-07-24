
# from sentence_transformers.cross_encoder import CrossEncoder

# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# query = "How many people live in Berlin?"
# passages = [
#     "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
#     "Berlin is well known for its museums.",
#     "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
#     "The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
#     "The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
#     "An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
#     "Berlin is subdivided into 12 boroughs or districts (Bezirke).",
#     "In 2015, the total labour force in Berlin was 1.85 million.",
#     "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
#     "Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
# ]

# ranks = model.rank(query, passages)

# print("Query", query)
# for rank in ranks:
#     print(f"{rank['score']:.2f}\t{passages[rank['corpus_id']]}")


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
"""
Company Guidelines RAG pipeline (fixed)
Hybrid retrieval (BM25 + dense FAISS) fused via weighted normalization,
re-ranked with a cross-encoder, answered by a local Qwen2.5 model.

Fixes applied vs. the original draft:
1. Dense scores are now computed for the WHOLE corpus (not just top-k),
   so they can be normalized/fused with BM25 scores of the same shape.
2. Retrieval indexing is consistent: everything indexes into the
   chunk-level `ids`/`texts` lists, not the parent-level `raw_docs`.
3. retrieve() has one clear contract:
     -> List[(chunk_id, text, hybrid_score)]
   so callers don't have to guess the shape.
4. Fixed `CROSS_ENCODER_MODEL.preidct` typo -> `.predict`.
5. Cross-encoder gets raw text, not dicts/tuples.
6. evaluate_retrieval / evaluate_generation updated to match the
   real return shape of retrieve() / answer_query().
"""

import json
# from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FILE_PATH = "companyGuidelines.txt"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

SPARSE_WEIGHT = 0.3
DENSE_WEIGHT = 0.3
RERANK_WEIGHT = 0.4

TOP_K_CANDIDATES = 20   # how many hybrid results go to the reranker
TOP_K_FINAL = 5

CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30

# ---------------------------------------------------------------------------
# Load + chunk data
# ---------------------------------------------------------------------------
with open(FILE_PATH, "r", encoding="utf-8") as f:
    guidelines = json.load(f)

raw_docs = [d for d in guidelines if d["id"] > 0]

def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
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


document_store = {}      # chunk_id -> chunk text
chunk_to_parent = {}      # chunk_id -> parent document id

for doc in raw_docs:
    for i, chunk in enumerate(chunk_text(doc["text"])):
        chunk_id = f"{doc['id']}_{i}"
        document_store[chunk_id] = chunk
        chunk_to_parent[chunk_id] = doc["id"]

ids = list(document_store.keys())
texts = list(document_store.values())

# ---------------------------------------------------------------------------
# Sparse retrieval (BM25) — scores over the WHOLE corpus
# ---------------------------------------------------------------------------
tokenized_corpus = [t.split(" ") for t in texts]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_scores_all(query: str) -> np.ndarray:
    return bm25.get_scores(query.split(" "))

# ---------------------------------------------------------------------------
# Dense retrieval (FAISS) — also scored over the WHOLE corpus
# ---------------------------------------------------------------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME)

doc_embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(doc_embeddings)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

def dense_scores_all(query: str) -> np.ndarray:
    """Cosine similarity against every doc — same shape as bm25_scores_all,
    so the two can be normalized/fused directly."""
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    return doc_embeddings @ q_emb[0]

# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def normalize_and_fuse(sparse_scores: np.ndarray, dense_scores: np.ndarray) -> np.ndarray:
    sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-10)
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-10)
    return SPARSE_WEIGHT * sparse_norm + DENSE_WEIGHT * dense_norm


# ---------------------------------------------------------------------------
# Retrieval + rerank — single, consistent contract:
#   returns List[(chunk_id, text, final_score)]
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K_FINAL):
    sparse_scores = bm25_scores_all(query)
    dense_scores = dense_scores_all(query)
    hybrid_scores = normalize_and_fuse(sparse_scores, dense_scores)

    # top candidates by hybrid score (indices into ids/texts)
    candidate_idx = np.argsort(hybrid_scores)[::-1][:TOP_K_CANDIDATES]

    # cross-encoder rerank on those candidates
    pairs = [(query, texts[i]) for i in candidate_idx]
    rerank_scores = CROSS_ENCODER_MODEL.predict(pairs, batch_size=32)

    final = []
    for i, idx in enumerate(candidate_idx):
        final_score = (
            (SPARSE_WEIGHT + DENSE_WEIGHT) * hybrid_scores[idx]
            + RERANK_WEIGHT * rerank_scores[i]
        )
        final.append((ids[idx], texts[idx], float(final_score)))

    final.sort(key=lambda x: x[2], reverse=True)
    return final[:top_k]


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)


def generate_answer(query: str, contexts: list[str]) -> str:
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
def answer_query(query: str, top_k: int = TOP_K_FINAL) -> str:
    top_chunks = retrieve(query, top_k=top_k)
    contexts = [text for _, text, _ in top_chunks]
    return generate_answer(query, contexts)


# ---------------------------------------------------------------------------
# Evaluation / metrics
# ---------------------------------------------------------------------------
# eval_set format:
# [{"query": "...", "relevant_ids": [3]}, ...]   (parent doc ids)

def evaluate_retrieval(eval_set, top_k=TOP_K_FINAL):
    hit_count = 0
    reciprocal_ranks = []
    precisions = []
    recalls = []

    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        retrieved = retrieve(query, top_k=top_k)  # List[(chunk_id, text, score)]
        retrieved_parent_ids = [chunk_to_parent[chunk_id] for chunk_id, _, _ in retrieved]

        hit = any(pid in relevant for pid in retrieved_parent_ids)
        hit_count += int(hit)

        rr = 0.0
        for rank, pid in enumerate(retrieved_parent_ids, start=1):
            if pid in relevant:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

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
    """Cheap smoke test: does the generated answer contain expected keyword(s)?
    eval_set items need an "expected_keywords" list."""
    correct = 0
    for item in eval_set:
        answer = answer_query(item["query"]).lower()
        if any(kw.lower() in answer for kw in item.get("expected_keywords", [])):
            correct += 1
    return correct / len(eval_set)


if __name__ == "__main__":
    query = "How to reset the password"
    print(answer_query(query))

    eval_set = [{"query": "How to reset the password", "relevant_ids": [1]}]
    print(evaluate_retrieval(eval_set))