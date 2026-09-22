
import json
import random

from pathlib import Path

from .chunk_sampler import (
    sample_single_chunks,
    sample_multi_chunk_groups,
)

from .question_generator import (
    QuestionGenerator,
)

from .question_validator import (
    QuestionValidator,
    is_valid,
)

from .deduplicator import (
    exact_deduplicate,
    semantic_deduplicate,
)

from .llm_client import LLMClient

from .config import (
    OVER_GENERATION_FACTOR
)

class DatasetBuilder:

    def __init__(self):

        self.generator = QuestionGenerator()
        self.validator = QuestionValidator()

        self.llm = LLMClient()

    def save_jsonl(self, questions, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:

            for question in questions:

                f.write(
                    json.dumps(
                        question.model_dump(),
                        ensure_ascii=False
                    )
                    + "\n"
                )

    def generate(self, chunks: list[dict], target: int) -> list:

        questions = []

        desired = int(target * OVER_GENERATION_FACTOR)

        while len(questions) < desired:

            remaining = desired - len(questions)

            batch_size = min(5, remaining)

            # ---------------------------------------------
            # Single chunk
            # ---------------------------------------------

            sampled = sample_single_chunks(chunks, batch_size)

            batch = self.generator.generate(
                chunks=sampled,
                num_questions=batch_size,
                question_type="single_chunk"
            )

            questions.extend(
                batch.questions
            )
    
            # ---------------------------------------------
            # Multi chunk
            # ---------------------------------------------

            groups = sample_multi_chunk_groups(
                chunks,
                max(1, batch_size // 2)
            )

            for group in groups:

                batch = self.generator.generate(
                    chunks=group,
                    num_questions=1,
                    question_type="multi_chunk"
                )

                questions.extend(
                    batch.questions
                )

            print(
                f"Generated: {len(questions)}"
            )

        return questions[:desired]

    def validate(
        self,
        questions: list,
        chunk_lookup: dict,
    ) -> list:

        valid = []

        for i, question in enumerate(
            questions
        ):

            source_chunks = []

            for item in question.relevant_chunks:

                chunk = chunk_lookup.get(
                    item.chunk_id
                )

                if chunk:
                    source_chunks.append(
                        chunk
                    )

            if not source_chunks:
                continue

            critique = self.validator.validate(
                question,
                source_chunks
            )

            if is_valid(critique):

                valid.append(question)

            print(
                f"Validated {i + 1}/{len(questions)} "
                f"| valid={len(valid)}"
            )

        return valid

    def deduplicate(
        self,
        questions: list,
    ) -> list:

        questions = exact_deduplicate(
            questions
        )

        print(
            f"After exact dedup: {len(questions)}"
        )

        texts = [
            q.question
            for q in questions
        ]

        embeddings = self.llm.embed(
            texts
        )

        questions = semantic_deduplicate(
            questions,
            embeddings
        )

        print(
            f"After semantic dedup: {len(questions)}"
        )

        return questions

    def split(
        self,
        questions: list,
        dev_size: int,
        test_size: int,
    ):

        random.shuffle(
            questions
        )

        test = questions[
            :test_size
        ]

        dev = questions[
            test_size:test_size + dev_size
        ]

        return dev, test

