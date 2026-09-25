"""Question definitions, in Laya's own wire format."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal["choice", "score", "noul"]

# A state is whatever the caller wants judged: plain text, a JSON object or a list of turns.
State = str | dict[str, Any] | list[Any]


class Question(BaseModel):
    """One typed question.

    - choice: `criteria` is {label: description | null} or a list of labels.
    - score:  `criteria` is a list of level descriptions, index 0 = lowest.
    - noul:   yes/no; `criteria` optionally {"true": "...", "false": "..."}.
    """

    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    instructions: str = Field(min_length=1, description="What the model should decide, e.g. 'Which team owns this bug?'")
    criteria: dict[str, str | None] | list[str] | None = Field(
        default=None, description="Labels (choice), ordered levels (score) or optional true/false descriptions (noul)"
    )

    def to_laya(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


Questions = dict[str, Question]


def questions_to_laya(questions: Questions) -> dict[str, dict[str, Any]]:
    """Convert validated questions into the plain dict `laya.Router.predict` expects."""
    return {qid: q.to_laya() for qid, q in questions.items()}
