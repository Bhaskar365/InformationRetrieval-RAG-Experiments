
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from transformers import Chunk
import numpy as np

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0,
    num_predict=1000,
)

def recall_at_k(
         retrieved_chunks: List[Chunk],
        relevant_documents: List[str],
        k: int
 ) -> float:

    retrieved_texts = [
        chunk.text
        for chunk in retrieved_chunks[:k]
    ]

    hits = sum(
        1 for document in relevant_documents if document in retrieved_texts
    )

    return hits/len(relevant_documents)



def precision_at_k(
        retrieved_chunks: List[Chunk],
        relevant_documents: List[str],
        k: int
) -> float:

    retrieved = retrieved_chunks[:k]

    relevant_count = sum(
        1 for chunk in retrieved if chunk.text if relevant_documents
    )

    return relevant_count / k



def reciprocal_rank(
        retrieved_chunks: List[Chunk],
        relevant_documents: List[str]
) -> float:

    for rank, chunk in enumerate(retrieved_chunks, start=1):

        if chunk.text in relevant_documents:
            return 1 / rank

    return 0.0


def semantic_similarity(
        answer: str,
        expected_answer: str
) -> float:

    embeddings = embed([
        answer,
        expected_answer
    ])

    return float(
        np.dot(
            embeddings[0],
            embeddings[1]
        )
    )


llm = ChatOllama(model='llama3.2:3b')

def evaluate_faithfulness(
        question: str,
        context: List[str],
        answer: str
) -> float:

    context_text = '\n'.join(context)

    prompt = f"""
You are evaluating a RAG system.

Determine whether the answer is fully supported
by the provided context.

Context:
{context_text}

Question:
{question}

Answer:
{answer}

Return ONLY a number from 0 to 1.

1.0 = completely supported
0.5 = partially supported
0.0 = unsupported
"""
    chain = prompt | llm | StrOutputParser()
    response = chain

    try:
        return float(response.output)
    except ValueError:
        return 0.0