
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

corpus = ['The sky is blue', 'The sun is bright', 'The grass is green']
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def hybrid_retrieve(query, top_k=2):
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    query_embedding = model.encode(query, convert_to_tensor=True)
    dense_scores = cosine_similarity([query_embedding.cpu().numpy()], corpus_embeddings.cpu().numpy())[0]

    combined_scores = 0.5 * np.array(bm25_scores) + 0.5 * np.array(dense_scores)
    ranked_indices = combined_scores.argsort()[::-1][:top_k]
    return [corpus[i] for i in ranked_indices]

print(hybrid_retrieve("blue sky"))

