"""Normalised answers returned by every decision tool."""
from typing import Literal

from pydantic import BaseModel, Field

from .questions import QuestionType

DecisionStatus = Literal["decided", "needs_review", "unverified"]
Detail = Literal["compact", "full"]


class Answer(BaseModel):
    type: QuestionType
    value: str | float | bool = Field(
        description="choice: chosen label; score: expected level (float); noul: true/false"
    )
    confidence: float = Field(description="Calibrated when the schema has a calibration, else as shipped")
    level: int | None = Field(default=None, description="score only: most likely level index")
    p_true: float | None = Field(default=None, description="noul only: probability of true")
    status: DecisionStatus | None = Field(
        default=None, description="Set by schema-based decisions: decided / needs_review / unverified"
    )
    probabilities: dict[str, float] | None = Field(default=None, description="detail=full only")
    legend: dict[str, str] | None = Field(default=None, description="score, detail=full only")


class ItemResult(BaseModel):
    index: int
    answers: dict[str, Answer]
    model: str = Field(description="Checkpoint that answered: english / multilingual / typed-decisions")
    route_reason: str | None = None
    needs_review: list[str] = Field(default_factory=list, description="Question ids below their confidence threshold")


class ClassifyResponse(BaseModel):
    results: list[ItemResult]
    rows: int = Field(description="items x questions evaluated")
    latency_ms: int
    warnings: list[str] = Field(default_factory=list)


class DecideResponse(ClassifyResponse):
    schema_ref: str = Field(description="team/name@version that was applied")
    schema_status: Literal["draft", "evaluated", "trusted"]
    decided: int = Field(description="answers at or above threshold")
    needs_review: int
