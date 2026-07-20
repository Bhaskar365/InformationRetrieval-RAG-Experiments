
from collections import defaultdict
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

try:
    
    query = "How to reset the password"
    filePath = 'companyGuidelines.txt'

    EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    TOP_K_DENSE = 5
    TOP_K_SPARSE = 5
    TOP_K_FINAL = 5

    with open(filePath,'r', encoding='utf-8') as f:
        guidelinesText = json.load(f)
        
    data = [ d for d in guidelinesText if d['id'] > 0 ]

    texts = [ doc['text'] for doc in data ]
    ids = [ d['id'] for d in data ]

    document_store = { doc["id"] : doc["text"] for doc in data }

    def rrf(result_lists, k=60):
        scores = defaultdict(float)

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                doc_id = result["id"]
                scores[doc_id] += 1/ (k + rank)

        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        return ranked

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    
    doc_embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(doc_embeddings)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
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

    tokenized_corpus = [doc.split(" ") for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(texts, convert_to_tensor=True)


    def bm25_search(query, top_k=TOP_K_SPARSE):

        tokenized_query = query.split(" ")
        scores = bm25.get_scores(tokenized_query)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [{"id": ids[i], "score": float(scores[i])} for i in ranked_idx]

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

    def answer_query(query, top_k=TOP_K_FINAL):
        dense_results = faiss_search(query)
        sparse_results = bm25_search(query)
        merged = rrf([dense_results, sparse_results])
        top_docs = merged[:top_k]
        contexts = [document_store[doc_id] for doc_id, _ in top_docs]
        return generate_answer(query, contexts)

    print(answer_query(query))

except Exception as e:
    print(e)
