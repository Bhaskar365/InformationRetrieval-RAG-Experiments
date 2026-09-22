

import json

from evaluation.config import (
    GENERATED_DIR,
    FINAL_DIR,
    DEV_SIZE,
    TEST_SIZE,
)

from evaluation.schemas import (
    EvaluationQuestion,
)

from evaluation.dataset_builder import (
    DatasetBuilder,
)


def load_questions(path):

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


def save_jsonl(
    questions,
    path,
):

    with open(path, "w", encoding="utf-8") as f:

        for q in questions:

            f.write(
                json.dumps(
                    q.model_dump(),
                    ensure_ascii=False
                )
                + "\n"
            )


def main():

    questions = load_questions(
        GENERATED_DIR
        / "validated_questions.jsonl"
    )

    print(
        f"Before dedup: {len(questions)}"
    )

    builder = DatasetBuilder()

    questions = builder.deduplicate(
        questions
    )

    print(
        f"After dedup: {len(questions)}"
    )

    # if len(questions) < (
    #     DEV_SIZE + TEST_SIZE
    # ):

    #     raise RuntimeError(
    #         "Not enough valid questions. "
    #         "Generate more."
    #     )

    dev, test = builder.split(
        questions,
        DEV_SIZE,
        TEST_SIZE
    )

    save_jsonl(
        dev,
        FINAL_DIR / "eval_dev.jsonl"
    )

    save_jsonl(
        test,
        FINAL_DIR / "eval_test.jsonl"
    )

    print(
        f"Dev: {len(dev)}"
    )

    print(
        f"Test: {len(test)}"
    )


if __name__ == "__main__":
    main()
