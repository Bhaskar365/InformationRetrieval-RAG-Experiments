
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

texts = [
    "Large language models require diverse datasets.",
    "Language models need large and diverse datasets.",
    "This is a duplicate sentence.",
    "This is a duplicate sentence."
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True
).astype('float32')

dimension = embeddings.shape[1]

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

generator = pipeline(
    "text-generation",
    model='Qwen/Qwen2.5-3B-Instruct',
    device_map='auto'
)

def llm_verify(text1, text2):

    prompt = f"""
You are a duplicate detection system.

Sentence A:
{text1}

Sentence B:
{text2}

Are these semantically duplicates?

Return ONLY JSON.

{{
    "duplicate": true,
    "confidence": 0.95,
    "reason": ""
}}
"""

    response = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    return response[0]["generated_text"]

# --------------------------
# Duplicate Detection
# --------------------------
threshold = 0.85

for i, text in enumerate(texts):

    query = embedding_model.encode(
        [text],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query)

    # retrieve 3 nearest neighbours
    distances, indices = index.search(query, 3)

    print("=" * 60)
    print("Current Record")
    print(text)

    for score, idx in zip(distances[0], indices[0]):

        # Skip self
        if idx == i:
            continue

        print(f"\nCandidate : {texts[idx]}")
        print(f"Similarity: {score:.3f}")

        if score > 0.95:

            print("Exact duplicate")

        elif score > threshold:

            print("Possible duplicate")
            print(llm_verify(text, texts[idx]))

        else:

            print("Different")




# def retrieval(query, k=3):
#     query_embedding=embedding_model.encode(
#         [query],
#         convert_to_numpy=True
#     )

#     matrix = cos_sim(embeddings, embeddings)
#     print(matrix)

#     faiss.normalize_L2(query_embedding)

#     distance, indices = index.search(
#         query_embedding,
#         k
#     )

#     results = []

#     for idx in indices[0]:
#         results.append(texts[idx])

#     for score, idx in zip(distance[0], indices[0]):
#         print(f"{score:.3f} -> {texts[idx]}")

#     if score > 0.95:
#         print("Duplicate")

#     elif score > 0.80:
#         print("Possible duplicate")
#         # Ask LLM

#     else:
#         print("Not duplicate")

#     return results

# def prompt(question):

#     context = retrieval(question)
#     context = '\n\n'.join(context)

#     prompt = f"""
# Sentence A:
# Large language models require diverse datasets.

# Sentence B:
# Language models need large and diverse datasets.

# Are these duplicates?

# Return JSON:
# {{
#  "duplicate": true,
#  "reason": "...",
#  "confidence": 0.91
# }}

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#     response = generator(
#         prompt,
#         max_new_tokens=200,
#         do_sample=False
#     )

#     return response[0]["generated_text"]

# print(
#     prompt(
#         "Language models require large datasets."
#     )
# )