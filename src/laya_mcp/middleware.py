"""FastMCP middleware: per-call usage recording and a schema-preserving response size limit.

`UsageMiddleware` times every tool call and records one usage row (team, tool, schema, rows,
answers, needs_review, latency, error flag) through `Repo.record_usage`. Counts come from the
tool's structured result (`rows`, `results[*].answers`, `results[*].needs_review`,
`schema_ref`); a tool whose result does not carry them (e.g. a batch submission) can report
them with `note_usage(...)` while it runs. Only counts and latencies are stored, never content.
Recording failures are logged and swallowed: usage stats must never break a call.
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from functools import partial
from typing import Any

import anyio
import pydantic_core
import structlog
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware
from fastmcp.tools.base import ToolResult

_notes: ContextVar[dict[str, Any] | None] = ContextVar("laya_usage_notes", default=None)
log = structlog.get_logger("laya_mcp.usage")


def note_usage(**fields: Any) -> None:
    """Called from inside a tool to report usage the structured result does not show
    (rows, schema_ref, answers, needs_review). No-op outside a recorded call."""
    notes = _notes.get()
    if notes is not None:
        notes.update({k: v for k, v in fields.items() if v is not None})


def _counts_from_result(structured: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(structured, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(structured.get("rows"), int):
        out["rows"] = structured["rows"]
    if isinstance(structured.get("schema_ref"), str):
        out["schema_ref"] = structured["schema_ref"]
    results = structured.get("results")
    if isinstance(results, list):
        answers = review = 0
        for r in results:
            if isinstance(r, dict):
                answers += len(r.get("answers") or {})
                review += len(r.get("needs_review") or [])
        out["answers"] = answers
        out["needs_review"] = review
    return out


class UsageMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        from .state import current_team, get_state

        tool = getattr(context.message, "name", None) or "unknown"
        notes: dict[str, Any] = {}
        token = _notes.set(notes)
        t0 = time.perf_counter()
        error = False
        result = None
        try:
            result = await call_next(context)
            error = bool(getattr(result, "is_error", False))
            return result
        except BaseException:
            error = True
            raise
        finally:
            _notes.reset(token)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            try:
                state = get_state()
                counts = _counts_from_result(getattr(result, "structured_content", None))
                counts.update(notes)
                args = getattr(context.message, "arguments", None) or {}
                schema_ref = counts.get("schema_ref")
                if schema_ref is None and isinstance(args.get("schema"), str):
                    schema_ref = args["schema"]
                record = partial(
                    state.repo.record_usage,
                    team=current_team(),
                    tool=tool,
                    schema_ref=schema_ref,
                    rows=int(counts.get("rows", 0)),
                    latency_ms=latency_ms,
                    answers=int(counts.get("answers", 0)),
                    needs_review=int(counts.get("needs_review", 0)),
                    error=error,
                )
                with anyio.CancelScope(shield=True):   # record cancelled calls too
                    await anyio.to_thread.run_sync(record)
            except Exception as e:  # never let usage recording break a call
                log.warning("usage_record_failed", tool=tool, error=str(e))


class StructuredResponseLimit(ResponseLimitingMiddleware):
    """ResponseLimitingMiddleware that keeps output schemas.

    The stock middleware hides every tool's outputSchema (it may truncate a response to plain text,
    which would then violate the schema). Laya tools return typed results that clients validate, so
    instead of truncating, an oversized *structured* response becomes a ToolError telling the agent
    how to ask for less. Unstructured (text-only) responses are truncated as usual.
    """

    async def on_list_tools(self, context, call_next):
        return await call_next(context)

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        if not isinstance(result, ToolResult) or result.structured_content is None:
            return await super().on_call_tool(context, _returning(result))
        if not self._limits_tool(context.message.name):
            return result
        size = len(pydantic_core.to_json(result, fallback=str))
        if size <= self.max_size:
            return result
        raise ToolError(
            f"The response ({size:,} bytes) exceeds the {self.max_size:,}-byte limit. Ask for less: a smaller "
            "`limit` (paginate with next_offset), detail='compact', fewer items or filters such as "
            "needs_review_only=true."
        )


def _returning(result: Any):
    async def call_next(_context):
        return result

    return call_next
