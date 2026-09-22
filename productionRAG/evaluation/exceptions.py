
import logging

logger = logging.getLogger(__name__)

"""Custom exceptions for the evaluation pipeline."""

class EvaluationError(Exception):
    """Base exception for evaluation pipeline."""
    print(f"Evaluation exception : {Exception}")
    logger.error(Exception)


class LLMError(EvaluationError):
    """LLM API or response errors."""
    print(f"LLM exception : {Exception}")
    logger.error(Exception)


class ValidationError(EvaluationError):
    """Question validation errors."""
    print(f"Validation exception : {Exception}")
    logger.error(Exception)


class GenerationError(EvaluationError):
    """Question generation errors."""
    print(f"Question generation exception : {Exception}")
    logger.error(Exception)


class RateLimiter(EvaluationError):
    """Rate limiting errors."""
    print(f"Rate limiting exception: {Exception}")
    logger.error(Exception)


class CheckpointError(EvaluationError):
    """Checkpoint errors."""
    print(f"Checkpoint exception: {Exception}")
    logger.error(Exception)