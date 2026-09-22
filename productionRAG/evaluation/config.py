
from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
EVAL_DIR = DATA_DIR / "evaluation"
GENERATED_DIR = EVAL_DIR / "generated"
FINAL_DIR = EVAL_DIR / "final"
RESULTS_DIR = EVAL_DIR / "results"

# LLM Configuration
GENERATION_MODEL = os.getenv("EVAL_LLM_GENERATION_MODEL", "llama3.2:3b")
CRITIC_MODEL = os.getenv("EVAL_CRITIC_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Generation Parameters
# TARGET_QUESTIONS = 400
TARGET_QUESTIONS = 30
GENERATION_BATCH_SIZE = 3
OVER_GENERATION_FACTOR = 5.0
MAX_GENERATION_ATTEMPTS = 2000
VALIDATION_BATCH_SIZE = 10

# Rate limiting
LLM_REQUESTS_PER_MINUTE = 30
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY_SECONDS = 2.0

# Quality Threshold
SEMANTIC_DUPLICATE_THRESHOLD = 0.90
MIN_QUALITY_SCORE = 0.80

# Question Distribution
QUESTION_DISTRIBUTION = {
    "single_chunk": 0.40,
    "multi_chunk": 0.25,
    "conceptual": 0.15,
    "comparison": 0.10,
    "reasoning": 0.10,
}

# data split
# DEV_SIZE = 250
# TEST_SIZE = 150
DEV_SIZE = 20
TEST_SIZE = 10

# Retrieval metrics
K_VALUES = [1, 3, 5, 10]

# Checkpointing
CHECKPOINT_INTERVAL = 50 # Save progress
ENABLE_CHECKPOINTING = True

for directory in [
    GENERATED_DIR,
    FINAL_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable configuration for question generation."""
    target_questions:int = TARGET_QUESTIONS
    over_generation_factor:float = OVER_GENERATION_FACTOR
    max_attempts:int = MAX_GENERATION_ATTEMPTS
    batch_size:int = GENERATION_BATCH_SIZE
    min_quality_score:float = MIN_QUALITY_SCORE
    checkpoint_interval:int = CHECKPOINT_INTERVAL
    enable_checkpointing:bool = ENABLE_CHECKPOINTING

    @property
    def desired_generation_count(self) -> int:
        return int(self.target_questions * self.over_generation_factor)