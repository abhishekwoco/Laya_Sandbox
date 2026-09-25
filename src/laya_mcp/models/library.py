"""Schemas (saved question sets), labeled datasets and evaluation reports."""
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .questions import Questions, QuestionType, State

SchemaStatus = Literal["draft", "evaluated", "trusted"]

_REF = re.compile(r"^(?P<team>[a-z0-9][a-z0-9_-]*)/(?P<name>[a-z0-9][a-z0-9_.-]*)(@(?P<version>\d+))?$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class SchemaRef(BaseModel):
    team: str
    name: str
    version: int | None = None   # None = latest

    @classmethod
    def parse(cls, ref: str, default_team: str | None = None) -> "SchemaRef":
        """Accept 'team/name', 'team/name@3', or bare 'name' when a default team is given."""
        ref = ref.strip().lower()
        if "/" not in ref and default_team:
            ref = f"{default_team}/{ref}"
        m = _REF.match(ref)
        if not m:
            raise ValueError(f"invalid schema reference {ref!r}; use team/name or team/name@version")
        v = m.group("version")
        return cls(team=m.group("team"), name=m.group("name"), version=int(v) if v else None)

    def __str__(self) -> str:
        return f"{self.team}/{self.name}" + (f"@{self.version}" if self.version else "")


def valid_name(name: str) -> bool:
    return bool(_NAME.match(name))


class SchemaTargets(BaseModel):
    min_accuracy: float = Field(default=0.9, ge=0, le=1, description="Accuracy every question must reach (at its threshold) to be promoted")
    min_examples: int = Field(default=50, ge=1)
    per_question: dict[str, float] = Field(default_factory=dict, description="Optional per-question accuracy overrides")


class SchemaInfo(BaseModel):
    team: str
    name: str
    version: int
    status: SchemaStatus
    description: str = ""
    questions: Questions
    targets: SchemaTargets
    thresholds: dict[str, float] = Field(default_factory=dict, description="Per-question confidence threshold; missing = server default")
    temperatures: dict[str, float] = Field(default_factory=dict, description="Per-question calibration temperature; missing = 1.0")
    latest_report_id: int | None = None
    created_at: datetime

    @property
    def ref(self) -> str:
        return f"{self.team}/{self.name}@{self.version}"


class SchemaSummary(BaseModel):
    ref: str
    status: SchemaStatus
    description: str
    question_ids: list[str]
    versions: int
    updated_at: datetime


ExpectedValue = str | int | float | bool


class LabeledExample(BaseModel):
    state: State
    expected: dict[str, ExpectedValue] = Field(
        description="question id -> correct answer: choice label, score level index (int), or noul true/false"
    )


class DatasetInfo(BaseModel):
    team: str
    name: str
    count: int
    question_ids: list[str]
    created_at: datetime
    updated_at: datetime


class ConfidenceBand(BaseModel):
    lo: float
    hi: float
    n: int
    accuracy: float | None


class LabelMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class QuestionMetrics(BaseModel):
    type: QuestionType
    n: int
    accuracy: float
    mae: float | None = None                 # score questions: |expected level - predicted expected score|
    ece: float
    temperature: float = 1.0                 # calibration applied when computing these numbers
    labels: list[str] = Field(default_factory=list)
    confusion: list[list[int]] = Field(default_factory=list, description="rows = expected label, cols = predicted, order = labels")
    per_label: dict[str, LabelMetrics] = Field(default_factory=dict)
    bands: list[ConfidenceBand] = Field(default_factory=list)
    recommended_threshold: float | None = None
    accuracy_at_threshold: float | None = None
    coverage_at_threshold: float | None = Field(default=None, description="Share of answers at or above the threshold")
    meets_target: bool = False


class Miss(BaseModel):
    example_index: int
    question_id: str
    expected: ExpectedValue
    predicted: ExpectedValue
    confidence: float


class EvalReport(BaseModel):
    id: int | None = None
    schema_ref: str
    dataset: str
    job_id: str | None = Field(default=None, description="Evaluate job whose raw results produced this report")
    created_at: datetime
    n_examples: int
    target_accuracy: float
    calibrated: bool
    per_question: dict[str, QuestionMetrics]
    overall_accuracy: float
    passes_targets: bool
    worst_misses: list[Miss] = Field(default_factory=list)
    model_counts: dict[str, int] = Field(default_factory=dict, description="Which checkpoint answered how many examples")
    notes: list[str] = Field(default_factory=list)


class CalibrationResult(BaseModel):
    schema_ref: str
    temperatures: dict[str, float]
    ece_before: dict[str, float]
    ece_after: dict[str, float]
    n_fit: int
    n_holdout: int
    report_id: int | None = Field(default=None, description="New evaluation report computed with the calibration applied")


def jsonable(obj: Any) -> Any:
    """Helper for storing Pydantic models in JSON columns."""
    return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
