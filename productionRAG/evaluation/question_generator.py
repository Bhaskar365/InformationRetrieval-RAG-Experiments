
from .llm_client import LLMClient
from .schemas import EvaluationBatch


SYSTEM_PROMPT = """
You are generating a benchmark dataset for evaluating
a Retrieval-Augmented Generation (RAG) system.

The benchmark will be used to measure retrieval quality.

STRICT RULES:

1. Questions must be answerable using ONLY the supplied chunks.

2. Do not use outside knowledge.

3. Every important claim in the ground-truth answer must
   be supported by the supplied chunks.

4. Identify the minimum chunks required to answer the question.

5. Do not copy sentences directly from the source.

6. Questions should resemble realistic user questions.

7. Avoid questions that are too trivial.

8. Avoid yes/no questions.

9. Avoid questions whose answer is not contained in the
   supplied context.

10. For multi-chunk questions, at least two chunks must
    contribute meaningful information.

11. Do not invent chunk IDs.

12. Return structured output only.
"""


def format_chunks(
    chunks: list[dict]
) -> str:

    parts = []

    for chunk in chunks:

        parts.append(
            f"""
CHUNK ID: {chunk["chunk_id"]}

PAPER ID: {chunk["document_id"]}

TITLE: {chunk.get("title", "")}

SECTION: {chunk.get("section", "")}

TEXT:
{chunk["text"]}
"""
        )

    return "\n".join(parts)


def build_prompt(
    chunks: list[dict],
    num_questions: int,
    question_type: str,
) -> str:

    context = format_chunks(chunks)

    return f"""
Generate {num_questions} evaluation questions.

Required question type:

{question_type}

SOURCE CHUNKS:

{context}

For every question:

- provide the question
- provide a ground-truth answer
- identify relevant chunk IDs
- classify difficulty
- explain why each relevant chunk is needed

The answer must be grounded entirely in the supplied chunks.

For single_chunk questions, one chunk should normally be sufficient.

For multi_chunk questions, the answer must require information
from at least two chunks.

Do not use information not contained in the supplied chunks.
"""


class QuestionGenerator:

    def __init__(self):

        self.llm = LLMClient()

    def generate(
        self,
        chunks: list[dict],
        num_questions: int,
        question_type: str,
    ) -> EvaluationBatch:

        prompt = build_prompt(
            chunks=chunks,
            num_questions=num_questions,
            question_type=question_type,
        )

        return self.llm.generate_questions(
            SYSTEM_PROMPT,
            prompt
        )

