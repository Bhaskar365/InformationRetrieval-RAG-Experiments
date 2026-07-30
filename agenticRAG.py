
import numpy as np

class VectorDatabase:

    def __init__(self):
        self.vectors = []

    def add_vector(self, vec_id, vector, metadata=None):
        record = {
            "id": vec_id,
            "vector": np.array(vector, dtype=np.float32),
            "metadata": metadata
        }

        self.vectors.append(record)

    def get_all_vectors(self):
        return self.vectors

    def _cosine_similarity(self, vec_a, vec_b):

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cos_sim = dot_product / (norm_a * norm_b + 1e-8)

        return cos_sim

    def search(self, query_vector, top_k=3):

        query_vector = np.array(query_vector, dtype=np.float32)
        results = []

        for record in self.vectors:
            sim = self._cosine_similarity(query_vector, record['vector'])

            results.append({
                'id': record['id'],
                'similarity': sim,
                'metadata': record['metadata']
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

db = VectorDatabase()

np.random.seed(10)

db.add_vector('vec1', np.random.rand(5), metadata= { 'text': 'First item' })
db.add_vector('vec2', np.random.rand(5), metadata= { 'text': 'Second item' })
db.add_vector('vec3', np.random.rand(5), metadata= { 'text': 'Third item' })
db.add_vector('vec4', np.random.rand(5), metadata= { 'text': 'Fourth item' })
db.add_vector('vec5', np.random.rand(5), metadata= { 'text': 'Fifth item' })
        
query_vector = np.random.rand(5)

print("Query vector", query_vector)

results = db.search(query_vector, top_k=3)
print(f"Query vector: {query_vector}\n")
for res in results:
    print(f"ID: {res['id']} | Similarity: {res['similarity']:.2f} | Metadata: {res['metadata']}")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

sentences = [
    "Kunal's cat is playing with wool.",
    "Ethan was born on 1st September 1700.",
    "Dogs are loyal animals.",
    "I love eating pizza but only on Fridays.",
    "Manika was born in Bhopal."
]

db = VectorDatabase()

for idx, sentence in enumerate(sentences):

    embedding = model.encode(sentence)

    db.add_vector(vec_id=f"sent_{idx}", vector=embedding, metadata={'sentence': sentence})

    query = "When is Ethan's birthday?"
    query_vec = model.encode(query)

    results = db.search(query_vec, top_k=3)
    print(f"Query: \"{query}\"\n")

    for res in results:
        print(f"Similar Sentence: {res['metadata']['sentence']}")
        print(f"Cosine Similarity Score: {res['similarity']:.2f}\n")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
llm_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def generate_rag_response(query):

    context = ""
    query_embedding = embedder.encode(query, convert_to_numpy=True).astype('float32')

    results = db.search(query_embedding, top_k=3)

    for res in results:
        context += f"{res['metadata']['sentence']}\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer accurately."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(llm_model.device)
    generated_ids = llm_model.generate(**model_inputs, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

print(generate_rag_response("When is Ethan's birthday?"))

