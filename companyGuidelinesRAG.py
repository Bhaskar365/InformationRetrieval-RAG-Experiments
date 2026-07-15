
from collections import defaultdict

query = "How to reset the password"


def rrf(result_lists, k=60):
    scores = defaultdict(float)

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            scores[doc_id] += 1/ (k + rank)

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return ranked

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

chunk_size = 300

def faiss_search(data):
    chunks = [ 
        data[i:i+chunk_size] for i in range(0, len(data), chunk_size) 
    ]

    embeddings = embedder.encode(chunks).astype('float32')
    faiss.normalize(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlat2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    query_embedding = embedder.encode(query)


dense_results = faiss_search(query)

sparse_results = bm25_search(query)

merged = rrf([dense_results, sparse_results])

top_docs = merged[:5]

contexts = [ document_store[doc_id] for doc_id, _ in top_docs ]

answer = llm.generate(query, contexts)
