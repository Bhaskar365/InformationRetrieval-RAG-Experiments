
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

reader = PdfReader("example.pdf")
text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

chunk_size = 500
overlap = 100

chunks = []
start = 0

while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start += chunk_size - overlap

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto"
)

def retreive(query, k=1):
    
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results

def ask_pdf(question):

    context = retreive(question)
    context = "\n\n".join(context)

    prompt = f"""
You are answering questions from a PDF.

Answer ONLY from the context below.

If the answer is not contained in the context, say:

"I could not find that information."

Context:
{context}

Question:
{question}

Answer:
"""
    response = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    return response[0]["generated_text"]

print(
    ask_pdf(
        "Tell me about the abstract."
    )
)

print(
    ask_pdf(
        "What methods were used?"
    )
)

print(
    ask_pdf(
        "Tell me about water."
    )
)
