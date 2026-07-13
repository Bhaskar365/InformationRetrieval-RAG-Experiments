
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open('shakespeare.txt', 'r') as f:
    text = f.read()

chunk_size = 200

chunks = [
    text[i:i+chunk_size] for i in range(0, len(text), chunk_size)
]

embeddings = embedder.encode(chunks)


import faiss
import numpy as np

dimension = embeddings.shape[1]

quantizer = faiss.IndexFlatL2(dimension)

nlist = 100

index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_L2
)

# index = faiss.IndexHNSWFlat(dimension)

index.train(embeddings.astype("float32"))

index.add(np.array(embeddings).astype("float32"))

query_embedding = embedder.encode(
    ["You are all resolved rather to die than to famish?"]
)

D, I = index.search(
    np.array(query_embedding).astype("float32"),
    k=3
)

context = "\n".join(
    chunks[i]
    for i in I[0]
)


prompt = f"""
You are given excerpts from Shakespeare.

From theese texts, generate new text that makes change sense today.

{context}

Question:

You are all resolved rather to die than to famish?

Answer:
"""

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

inputs = tokenizer(
    prompt,
    return_tensors='pt'
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.2,
    do_sample=False
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(answer)


