
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

documents = [
    "Python is a programming language.",
    "Machine learning uses data to train models.",
    "Deep learning is a subset of machine learning.",
    "Cats are popular household pets.",
    "Dogs are loyal animals.",
    "Neural networks are inspired by the human brain.",
    "Football is a popular sport.",
    "Transformers are used in modern NLP systems.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

print("Embedding shape:", embeddings.shape)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored: ", index.ntotal)

def search(query, k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    print("\nQuery:", query)
    print("-" * 50)

    for rank, idx in enumerate(indices[0], start=1):
        print(f"{rank}. {documents[idx]}")
        print(f"   Distance: {distances[0][rank-1]:.4f}")


search("artificial intelligence and neural networks")

search("pets that live in houses")

search("soccer game")