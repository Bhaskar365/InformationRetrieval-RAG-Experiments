
from .llm_client import LLMClient
from .schemas import (
    EvaluationQuestion,
    Critique,
)


# SYSTEM_PROMPT = """
# You are a strict evaluator validating a RAG benchmark question.

# Determine whether:

# 1. The question is answerable from the supplied chunks.

# 2. The ground-truth answer is fully supported.

# 3. The listed relevant chunks are actually relevant.

# 4. Important supporting chunks are missing.

# 5. The answer contains outside knowledge.

# 6. The question is useful for evaluating retrieval.

# Be strict.

# Reject questions that are ambiguous, trivial, unsupported,
# or dependent on external knowledge.

# IMPORTANT:
# The quality_score MUST be a floating-point number between 0.0 and 1.0.

# Use this scale:

# 0.0 = completely unusable
# 0.25 = poor
# 0.50 = mediocre
# 0.75 = good
# 1.0 = excellent

# Do not use a 0-10 scale.
# For example, return 0.85, not 8.5.
# """

SYSTEM_PROMPT = """
You are a strict evaluator validating a RAG benchmark question.

Evaluate the question using ONLY the supplied source context.

Determine:

1. answerable:
   True if the question can be answered using the supplied chunks.
   False otherwise.

2. answer_supported:
   True if the ground-truth answer is fully supported by the supplied chunks.
   False if important claims are unsupported.

3. chunks_correct:
   True if all declared relevant chunks genuinely support answering
   the question.

4. missing_information:
   True if important information required to answer the question
   is missing from the supplied context.

5. outside_knowledge:
   True if the ground-truth answer requires information not present
   in the supplied chunks.

6. quality_score:
   A number between 0.0 and 1.0 representing the overall quality
   of the benchmark question.

   0.0 = unusable
   0.5 = mediocre
   0.8 = good
   1.0 = excellent

7. reason:
   Briefly explain your evaluation.

Be strict.

Reject questions that are ambiguous, trivial, unsupported,
or dependent on external knowledge.

IMPORTANT:
quality_score MUST be between 0.0 and 1.0.
NEVER return a score between 0 and 10.
"""



def build_validation_prompt(
    question: EvaluationQuestion,
    chunks: list[dict],
) -> str:

    context = "\n\n".join(
        f"""
CHUNK ID: {c["chunk_id"]}

TEXT:
{c["text"]}
"""
        for c in chunks
    )

    return f"""
QUESTION:

{question.question}


GROUND TRUTH ANSWER:

{question.ground_truth_answer}


DECLARED RELEVANT CHUNKS:

{
    [x.chunk_id for x in question.relevant_chunks]
}


SOURCE CONTEXT:

{context}

Evaluate this benchmark question.
"""


class QuestionValidator:

    def __init__(self):

        self.llm = LLMClient()

    def validate(
        self,
        question: EvaluationQuestion,
        chunks: list[dict],
    ) -> Critique:

        prompt = build_validation_prompt(
            question,
            chunks
        )

        return self.llm.critique_question(
            SYSTEM_PROMPT,
            prompt
        )


def is_valid(
    critique: Critique,
) -> bool:

    return (
        critique.answerable
        and critique.answer_supported
        and critique.chunks_correct
        and not critique.missing_information
        and not critique.outside_knowledge
        and critique.quality_score >= 0.80
    )
