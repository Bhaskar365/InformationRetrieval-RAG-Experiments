

from evaluation.config import (
    CHUNKS_FILE,
    GENERATED_DIR,
)

from evaluation.chunk_loader import (
    load_chunks,
    build_chunk_lookup,
)

from evaluation.schemas import (
    EvaluationQuestion,
)

from evaluation.dataset_builder import (
    DatasetBuilder,
)


def load_questions(path):

    import json

    questions = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            questions.append(
                EvaluationQuestion.model_validate(
                    json.loads(line)
                )
            )

    return questions


def main():

    chunks = load_chunks(
        CHUNKS_FILE
    )

    chunk_lookup = build_chunk_lookup(
        chunks
    )

    questions = load_questions(
        GENERATED_DIR / "raw_questions.jsonl"
    )

    builder = DatasetBuilder()

    valid = builder.validate(
        questions,
        chunk_lookup
    )

    output = (
        GENERATED_DIR
        / "validated_questions.jsonl"
    )

    builder.save_jsonl(
        valid,
        output
    )

    print(
        f"Valid questions: {len(valid)}"
    )


if __name__ == "__main__":
    main()

