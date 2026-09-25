"""FastAPI host: /mcp (FastMCP streamable HTTP), /health, /metrics, /v1/systemone.

Run with `laya-mcp` (see pyproject scripts) or `python -m laya_mcp.app`.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastmcp.utilities.lifespan import combine_lifespans

from . import __version__
from .config import Settings, get_settings
from .engine.runtime import BudgetExceeded, EngineBusy, InvalidQuestions
from .logging import configure_logging, get_logger
from .server import build_server
from .state import AppState, get_state, set_state

log = get_logger("laya_mcp.app")


def create_app(settings: Settings | None = None, *, engine: Any = None, repo: Any = None, jobs: Any = None) -> FastAPI:
    """Build the HTTP app. `engine`/`repo`/`jobs` may be injected (tests); otherwise they are built
    from `settings` when the app starts, not here, so creating the app stays cheap."""
    settings = settings or get_settings()
    mcp = build_server(settings)
    mcp_app = mcp.http_app(
        path="/mcp",
        stateless_http=True,
        json_response=True,
        # Intranet clients connect as http://10.10.29.81:8765/mcp or by hostname: never reject a Host.
        host_origin_protection=False,
    )

    @asynccontextmanager
    async def laya_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        eng, rep, js = engine, repo, jobs
        if eng is None:
            from .engine.runtime import InferenceEngine

            eng = InferenceEngine(settings)
        if rep is None:
            from .db.repo import Repo

            rep = Repo(settings.db_url)
        await anyio.to_thread.run_sync(rep.migrate)
        if js is None:
            from .jobs.service import JobService

            js = JobService(settings, rep, eng)
        await eng.start()
        set_state(AppState(settings=settings, engine=eng, repo=rep, jobs=js))
        try:
            from .workbench.hooks import register_hooks

            register_hooks(js)          # before jobs.start(): re-enqueued evaluate jobs need their hook
            await anyio.to_thread.run_sync(js.start)
            log.info("laya_mcp_started", version=__version__, host=settings.host, port=settings.port,
                     loaded=eng.status().loaded)
            yield
        finally:
            try:
                await anyio.to_thread.run_sync(js.stop)
            finally:
                try:
                    await eng.stop()
                finally:
                    set_state(None)
                    log.info("laya_mcp_stopped")

    app = FastAPI(
        title="laya-mcp",
        version=__version__,
        summary="Laya decisions for coding agents over MCP (/mcp) and the Jev /v1/systemone protocol",
        lifespan=combine_lifespans(laya_lifespan, mcp_app.lifespan),
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            st = get_state()
        except RuntimeError:
            return JSONResponse({"status": "starting", "version": __version__}, status_code=503)
        body: dict[str, Any] = {"status": "ok", "version": __version__}
        try:
            body["engine"] = st.engine.status().model_dump(mode="json")
        except Exception as e:
            body["status"], body["engine"] = "degraded", {"error": str(e)}
        try:
            await anyio.to_thread.run_sync(lambda: st.repo.list_schemas(team="__health__"))
            body["db"] = "ok"
        except Exception as e:
            body["status"], body["db"] = "degraded", f"error: {e}"
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Any:
        """Jev-compatible decision endpoint (same contract as laya.serve), routed through the shared
        engine so it respects the single-inference lock, limits and timeouts."""
        from laya.serve import _resolve_model

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        questions = body["questions"]
        if not isinstance(questions, dict) or not questions:
            raise HTTPException(status_code=422, detail="'questions' must be a non-empty object {id: question}")
        state = body.get("state")
        model = _resolve_model(body.get("model"))
        st = get_state()
        try:
            st.engine.check_sync_budget(1, len(questions), [state] if state is not None else None)
            results = await st.engine.predict(
                [state], questions, model=model, timeout=st.settings.request_timeout_s
            )
        except EngineBusy as e:
            return JSONResponse(
                {"detail": str(e)}, status_code=503, headers={"Retry-After": str(max(1, round(e.retry_after_s)))}
            )
        except (BudgetExceeded, InvalidQuestions) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except TimeoutError:
            raise HTTPException(status_code=504, detail=f"no answer within {st.settings.request_timeout_s:.0f}s")
        except Exception as e:  # noqa: BLE001 - laya surfaces bad questions/tokenizer errors as ValueError
            raise HTTPException(status_code=422, detail=str(e))
        # Laya's result is already Jev-shaped: {model, answers, usage, routing}.
        return results[0]

    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )

    # Last: everything not matched above goes to the MCP app, which serves /mcp.
    app.mount("/", mcp_app)
    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,          # our structlog config handles uvicorn's loggers
        workers=1,                # one process: the model and the inference lock are shared in memory
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
