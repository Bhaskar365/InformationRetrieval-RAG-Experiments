
import faiss

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
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
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map='auto'
)

def checkDuplicate(query,k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # matrix = cos_sim(embeddings, embeddings)
    # print(matrix)

    faiss.normalize_L2(query_embedding)

    distance, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(distance[0], indices[0]):
        print(f"{score:.3f} -> {texts[idx]}")
    
        results.append({
            "text" : texts[idx],
            "score" : float(score)
        })
    
    best_score = float(distance[0][0])
    best_text = texts[indices[0][0]]

    if best_score >= 0.95:
        decision = "Duplicate"
        print("Duplicate")

    elif best_score >= 0.80:
        decision = "Possible duplicate"
        print("Possible duplicate")

    else:
        decision = "Not duplicate"
        print("Not duplicate")

    # return results

    return {
        "query" : query,
        "best match" : best_text,
        "similarity" : best_score,
        "decision" : decision,
        "neighbors" : results
    }

queries = "This is a duplicate sentence."
    

result = checkDuplicate(queries)
print(result)


# def prompt(question):
#     context = checkDuplicate(question)
#     context = '\n\n'.join(context)


#     prompt = f"""
# Sentence A:
# Large language models require diverse datasets.

# Sentence B:
# Language models need large and diverse datasets.

# Are these duplicates?

# Return JSON:
# {{
#     "duplicate":true,
#     "reason":"",
#     "confidence":0.91
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

# print(prompt('This is a duplicate sentence'))



