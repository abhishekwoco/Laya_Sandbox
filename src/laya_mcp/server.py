"""The main FastMCP server: instructions, middleware stack, mounted tool sub-servers, resources, prompts."""
from __future__ import annotations

import logging

from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from . import __version__, prompts, resources
from .config import Settings, get_settings
from .middleware import StructuredResponseLimit, UsageMiddleware

MAX_RESPONSE_BYTES = 800_000
SEC_PER_ROW_HINT = "1"   # measured on this host; see docs/perf.md

INSTRUCTIONS = """\
Laya is a fast, calibrated classifier (a "System 1") shared by every team on this network. Delegate
mechanical decisions to it the way you would hand work to a subagent: you send content plus typed
questions, Laya answers all of them in one forward pass with a confidence for each, and you only spend
your own reasoning on the answers it is unsure about.

Use Laya for: routing and tagging (which team, which component, which category), triage (urgency,
severity, sentiment), yes/no checks over many items (does this log line show a timeout? does this
issue include a repro?), screening untrusted text for prompt injection (laya_scan_untrusted).
Do not use it for anything that needs reasoning, arithmetic, code generation or world knowledge.

How to call it well:
- Send concise, pre-extracted content (a title, the first lines of a stack trace, a diff hunk), never
  whole files: the model reads ~512 tokens.
- Ask all questions about an item in one call. Cost: rows = items x questions, about {sec_per_row} s per
  row on this CPU host (laya_status shows the live figure). Synchronous calls are capped at
  {sync_row_budget} rows. Above that use laya_classify_batch and poll laya_job_status.
- Recurring decisions belong in a saved schema (laya_save_schema -> laya_evaluate -> laya_calibrate ->
  laya_promote_schema); laya_decide then applies thresholds measured on labeled data.
- Answers with status 'needs_review' (or below your threshold) are yours to decide: Laya is telling you
  it is unsure. 'decided' answers (trusted schemas only) can be used as-is. 'unverified' means the
  schema has not been promoted to trusted yet: treat those answers as suggestions.
- Read laya://docs/question-types for the question format and how to write good criteria.
"""


def _core_servers() -> list[FastMCP]:
    from .tools import batch, decide, guard, ops

    return [decide.server, batch.server, guard.server, ops.server]


def _library_servers() -> list[FastMCP]:
    from .tools import library, workbench

    return [library.server, workbench.server]


def build_server(settings: Settings | None = None) -> FastMCP:
    settings = settings or get_settings()
    mcp = FastMCP(
        "laya",
        instructions=INSTRUCTIONS.format(
            sync_row_budget=settings.sync_row_budget, sec_per_row=SEC_PER_ROW_HINT
        ),
        version=__version__,
        middleware=[
            # transform_errors=False: ToolErrors must reach the client as isError results carrying our
            # message (with the fix), not be rewrapped as JSON-RPC "Internal error".
            ErrorHandlingMiddleware(logger=logging.getLogger("laya_mcp.errors"), transform_errors=False),
            StructuredLoggingMiddleware(
                logger=logging.getLogger("laya_mcp.requests"), include_payloads=settings.log_content
            ),
            TimingMiddleware(logger=logging.getLogger("laya_mcp.timing")),
            UsageMiddleware(),
            StructuredResponseLimit(max_size=MAX_RESPONSE_BYTES),
        ],
    )
    for sub in _core_servers() + _library_servers():
        mcp.mount(sub)
    resources.register(mcp)
    prompts.register(mcp)
    return mcp
