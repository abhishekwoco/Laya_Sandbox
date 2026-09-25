"""Helpers shared by the tool sub-servers: error mapping, state/items handling, usage notes.

Any tool module may use these; they only depend on the shared contracts.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterator, Literal, TypeVar

import anyio
from fastmcp.exceptions import ToolError

from ..db.repo import NotFound
from ..engine.runtime import BudgetExceeded, EngineBusy, InvalidQuestions
from ..models import ItemResult, State

T = TypeVar("T")

ModelName = Literal["english", "multilingual", "typed-decisions"]

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
WRITES = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}


@contextmanager
def tool_errors(timeout_s: float | None = None) -> Iterator[None]:
    """Translate engine/repo errors into ToolErrors whose message tells the agent what to do."""
    try:
        yield
    except ToolError:
        raise
    except EngineBusy as e:
        raise ToolError(f"{e} [retryable: retry_after_s={e.retry_after_s:.0f}]") from e
    except BudgetExceeded as e:
        raise ToolError(f"Request too large: {e}") from e
    except InvalidQuestions as e:
        raise ToolError(
            f"Invalid questions: {e}. Fix the question and retry (laya_validate_questions checks a "
            "question set without running the model; laya://docs/question-types explains the format)."
        ) from e
    except NotFound as e:
        raise ToolError(f"Not found: {e}") from e
    except TimeoutError as e:
        limit = f" within {timeout_s:.0f}s" if timeout_s else ""
        raise ToolError(
            f"Laya did not answer{limit}. Send fewer items or questions, shorten the content, "
            "or use laya_classify_batch for large workloads."
        ) from e
    except ValueError as e:
        raise ToolError(str(e)) from e


def resolve_states(state: State | None, items: list[State] | None) -> list[State]:
    """Exactly one of `state` / `items` must be given."""
    if state is not None and items is not None:
        raise ToolError("Pass either `state` (one item) or `items` (several), not both.")
    if items is not None:
        if not items:
            raise ToolError("`items` is empty; pass at least one item.")
        return list(items)
    if state is None:
        raise ToolError("Nothing to judge: pass `state` (one item) or `items` (a list of items).")
    return [state]


def state_chars(state: State) -> int:
    return len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))


def count_review(results: list[ItemResult]) -> tuple[int, int]:
    """(answers, answers flagged needs_review) across results."""
    answers = sum(len(r.answers) for r in results)
    review = sum(len(r.needs_review) for r in results)
    return answers, review


async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking call (repo, tokenizer load) off the event loop."""
    return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)
