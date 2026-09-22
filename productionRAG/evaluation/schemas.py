
from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum

QuestionType = Literal[
    "single_chunk",
    "multi_chunk",
    "conceptual",
    "comparison",
    "reasoning",
]

Difficulty = Literal["easy", "medium", "hard"]
RelevanceType = Literal["primary", "supporting"]

class GenerationStatus(str,Enum):
    PENDING = "pending"
    GENERATED = "generated"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FAILED = "failed"

class RelevantChunk(BaseModel):
    chunk_id: str
    relevance: RelevanceType
    rationale: str

# Original
# class EvaluationQuestion(BaseModel):

#     question_id: str
#     question: str
#     ground_truth_answer: str
#     question_type: QuestionType
#     difficulty: Difficulty
#     relevant_chunks: List[RelevantChunk]
#     paper_ids: List[str]
#     answerable_from_context: bool
#     generation_metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluationQuestion(BaseModel):
    question_id: str
    question: str
    ground_truth_answer: str
    question_type: QuestionType
    difficulty: Difficulty
    relevant_chunks: List[RelevantChunk]
    paper_ids: List[str]
    answerable_from_context: bool
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at:datetime = Field(default_factory=datetime.utcnow)
    status: GenerationStatus = GenerationStatus.PENDING

class Config:
    json_encoders = {
        datetime: lambda v: v.isoformat()
    }   

class EvaluationBatch(BaseModel):
    questions: List[EvaluationQuestion]


class Critique(BaseModel):
    answerable: bool
    answer_supported: bool
    chunks_correct: bool
    missing_information: bool
    outside_knowledge: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls,v:float)->float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"quality_score must be between 0.0 and 1.0, got {v}")
        return v

class CritiqueBatch(BaseModel):
    critiques: List[Critique]


class GenerationMetrics(BaseModel):
    """Track generation pipeline metrics."""
    total_attempted:int = 0
    total_generated:int = 0
    total_valid:int = 0
    total_rejected:int = 0
    total_failed:int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time:Optional[datetime] = None
