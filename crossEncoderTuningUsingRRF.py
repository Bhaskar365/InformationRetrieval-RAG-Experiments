
import json
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


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



def retrieve(query:str, top_k:int=TOP_K_FINAL):
    sparse_scores = bm25_scores_all(query)
    dense_scores = dense_scores_all(query)

    # hybrid_scores = normalize_and_fuse(sparse_scores, dense_scores)

    sparse_ranked = scores_to_ranked(sparse_scores, ids)
    dense_ranked = scores_to_ranked(dense_scores, ids)

    fused = rrf([sparse_ranked, dense_ranked])
    fused_scores = dict(fused)

    candidate_ids = [chunk_id for chunk_id, _ in fused[:TOP_K_CANDIDATES]]

    pairs = [(query, document_store[cid]) for cid in candidate_ids]
    rerank_scores = CROSS_ENCODER_MODEL.predict(pairs, batch_size=32)

    final = []
    for i,chunk_id in enumerate(candidate_ids):
        final_score = (
            (SPARSE_WEIGHT + DENSE_WEIGHT) * fused_scores[chunk_id] + RERANK_WEIGHT * rerank_scores[i]
        )

        final.append((chunk_id, document_store[chunk_id], float(final_score)))

    final.sort(key=lambda x:x[2], reverse=True)
    return final[:top_k]

def scores_to_ranked(scores: np.ndarray, ids: list) -> list[dict]:
    """Convert a score array into a rank-ordered list of {'id': chunk_id} dicts,
    which is what rrf() expects."""

    ranked_ids = np.argsort(scores)[::-1]
    return [{ 'id': ids[i]} for i in ranked_ids]

def rrf(result_list, k=60):
    scores = defaultdict(float)
    for results in result_list:
        for rank,result in enumerate(results, start=1):
            scores[result['id']] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x:x[1], reverse=True)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)

def generate_answer(query:str, contexts: list[str]) -> str:
    context_block = '\n'.join(contexts)

    messages = [
        {"role": "system", "content": "Answer using ONLY the guideline text below if it answers the question. Do not add commentary or notes. If nothing below answers the question, output only: No such rule found."},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion:\n{query}"},
]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = llm_model.generate(**inputs, max_new_tokens=300, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def answer_query(query:str, top_k:int=TOP_K_FINAL) -> str:
    top_chunks = retrieve(query, top_k=top_k)
    contexts = [text for _, text, _ in top_chunks]
    return generate_answer(query, contexts)

def evaluate_retrieval(eval_set, top_k=TOP_K_FINAL):
    hit_count = 0
    reciprocal_ranks = []
    precisions = []
    recalls = []

    for item in eval_set:
        query = item['query']
        relevant = set(item['relevant_ids'])

        retrieved = retrieve(query,top_k=top_k)
        retrieved_parent_ids = [chunk_to_parent[chunk_id] for chunk_id, _, _ in retrieved]

        hit = any(pid in relevant for pid in retrieved_parent_ids)
        hit_count += int(hit)

        rr = 0.0
        for rank, pid in enumerate(retrieved_parent_ids, start=1):
            if pid in relevant:
                rr = 1/rank
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

if __name__ == '__main__':
    query = 'How to reset the password'
    print(answer_query(query))

    eval_set = [{"query": "How to reset the password", "relevant_ids": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}]
    print(evaluate_retrieval(eval_set))
    
