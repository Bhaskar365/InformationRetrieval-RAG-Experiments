
import logging

from evaluation.config import (
    CHUNKS_FILE,
    GENERATED_DIR,
    TARGET_QUESTIONS
)

from evaluation.chunk_loader import (
    load_chunks,
)

from evaluation.dataset_builder import (
    DatasetBuilder,
)

logger = logging.getLogger(__name__)

def main():

    try:
        print("Loading chunks...")
        logger.warning("Started loading chunks")

        chunks = load_chunks(
            CHUNKS_FILE
        )

        print(
            f"Loaded {len(chunks)} chunks"
        )

        logger.warning(f"Loaded {len(chunks)} chunks")

        builder = DatasetBuilder()

        print("Generating questions...")
        logger.warning("Started Generating questions...")

        questions = builder.generate(
            chunks=chunks,
            target=TARGET_QUESTIONS
        )

        output = (
            GENERATED_DIR/"raw_questions.jsonl"
        )

        builder.save_jsonl(
            questions,
            output
        )

        print(
            f"Saved {len(questions)} questions"
        )
        logger.info(f"Saved {len(questions)} questions")

        print(
            f"Output: {output}"
        )
        logger.info(f"Output: {len(output)}")

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

