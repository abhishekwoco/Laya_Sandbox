"""Library + workbench tools through the FastMCP in-memory client.

Every test runs twice: against the in-memory FakeRepo/FakeJobs (tests/fakes_workbench.py) and
against the real SQLite Repo + JobService (Huey immediate mode, so jobs finish inside submit and
fire the evaluate hook). Inference is FakeEngine (tests/fakes.py) in both."""
from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from laya_mcp.config import Settings
from laya_mcp.state import AppState, set_state
from laya_mcp.tools import library, workbench
from laya_mcp.workbench.hooks import register_hooks
from tests.fakes import FakeEngine
from tests.fakes_workbench import FakeJobs, FakeRepo

QUESTIONS = {
    "area": {
        "type": "choice",
        "instructions": "Which area is affected?",
        "criteria": {"backend": "server database", "frontend": "button layout"},
    },
    "bug": {"type": "noul", "instructions": "Is this a bug"},
}

TEMPLATES = [
    ("the database query crashes, a real bug", "backend", True),
    ("please add a database index for speed", "backend", False),
    ("the button renders wrong, a bug", "frontend", True),
    ("request to change the layout colours", "frontend", False),
]

LIBRARY_TOOLS = {"laya_save_schema", "laya_get_schema", "laya_list_schemas", "laya_promote_schema", "laya_get_report"}
WORKBENCH_TOOLS = {
    "laya_validate_questions", "laya_save_dataset", "laya_list_datasets", "laya_evaluate", "laya_calibrate",
    "laya_compare_models", "laya_sample_for_labeling",
}


def examples(n: int, flip_every: int = 0) -> list[dict]:
    out = []
    for i in range(n):
        text, area, bug = TEMPLATES[i % 4]
        if flip_every and i % flip_every == 0:
            bug = not bug
        out.append({"state": f"ticket {i}: {text}", "expected": {"area": area, "bug": bug}})
    return out


@pytest.fixture(params=["fakes", "real"])
def env(request, tmp_path):
    """'fakes': in-memory FakeRepo/FakeJobs. 'real': SQLite Repo + JobService in Huey immediate mode."""
    settings = Settings(
        data_dir=tmp_path, preload="english", warmup=False, threads=None,
        min_eval_examples=10, max_items_per_call=50, default_team="dev",
    )
    engine = FakeEngine(settings)
    if request.param == "fakes":
        repo, jobs = FakeRepo(), None
        jobs = FakeJobs(repo, engine)
    else:
        from laya_mcp.db.repo import Repo
        from laya_mcp.jobs.service import JobService

        repo = Repo(settings.db_url)
        repo.migrate()
        jobs = JobService(settings, repo, engine, immediate=True)
        jobs.start()
    register_hooks(jobs)
    state = AppState(settings=settings, engine=engine, repo=repo, jobs=jobs)
    set_state(state)
    yield state
    set_state(None)
    if request.param == "real":
        jobs.stop()
        repo.close()


@pytest.fixture
async def client(env):
    app = FastMCP("laya-test")
    app.mount(library.server)
    app.mount(workbench.server)
    async with Client(app) as c:
        yield c


async def call(c: Client, tool: str, /, **args):
    res = await c.call_tool(tool, args)
    return res.structured_content


async def save_schema(c, name="triage", **kw):
    targets = kw.pop("targets", {"min_accuracy": 0.9, "min_examples": 10})
    return await call(c, "laya_save_schema", name=name, questions=QUESTIONS, targets=targets, **kw)


async def evaluate(c, schema="dev/triage", dataset="dev/issues"):
    started = await call(c, "laya_evaluate", schema=schema, dataset=dataset)
    job = started["job"]
    assert job["status"] == "completed" and "report_id" in job["result_ref"], job
    return started, job["result_ref"]["report_id"]


# --------------------------------------------------------------------------- listing
async def test_tools_listed_with_annotations(client):
    tools = {t.name: t for t in await client.list_tools()}
    assert LIBRARY_TOOLS | WORKBENCH_TOOLS <= set(tools)
    for name in ("laya_get_schema", "laya_list_schemas", "laya_get_report", "laya_validate_questions",
                 "laya_list_datasets", "laya_compare_models", "laya_sample_for_labeling"):
        assert tools[name].annotations.read_only_hint is True, name
    for name in ("laya_save_schema", "laya_promote_schema", "laya_evaluate", "laya_calibrate", "laya_save_dataset"):
        assert tools[name].annotations.read_only_hint is False, name
    assert "laya_calibrate" in tools["laya_save_schema"].description   # teaches the workflow
    assert tools["laya_evaluate"].output_schema is not None


# --------------------------------------------------------------------------- validate
async def test_validate_questions(client):
    ok = await call(client, "laya_validate_questions", questions=QUESTIONS)
    assert ok["valid"] and ok["errors"] == [] and ok["rows_per_state"] == 2
    assert ok["est_seconds_per_state"] >= 0

    bad = await call(client, "laya_validate_questions", questions={
        "a": {"type": "bogus", "instructions": "x"},
        "b": {"type": "score", "instructions": "How bad?", "criteria": {"low": None}},
        "c": {"type": "choice", "instructions": "Pick"},
        "d": {"type": "noul", "instructions": "Fine?"},
    })
    assert not bad["valid"]
    joined = "\n".join(bad["errors"])
    assert "question 'a'" in joined and "question 'b'" in joined and "question 'c'" in joined
    assert "'d'" not in joined


# --------------------------------------------------------------------------- schemas
async def test_save_get_list_schema_versions(client):
    v1 = await save_schema(client, description="first")
    assert v1["schema_info"]["version"] == 1 and v1["schema_info"]["status"] == "draft"
    assert "laya_save_dataset" in v1["next_step"]
    urgent = {"urgent": {"type": "noul", "instructions": "Is it urgent"}}
    v2 = await call(client, "laya_save_schema", name="dev/triage", questions={**QUESTIONS, **urgent},
                    description="second")
    assert v2["schema_info"]["version"] == 2

    latest = await call(client, "laya_get_schema", schema="triage")
    assert latest["version"] == 2 and set(latest["questions"]) == {"area", "bug", "urgent"}
    first = await call(client, "laya_get_schema", schema="dev/triage@1")
    assert first["version"] == 1 and set(first["questions"]) == {"area", "bug"}

    listed = await call(client, "laya_list_schemas")
    [summary] = listed["schemas"]
    assert listed["team"] == "dev" and summary["ref"].startswith("dev/triage") and summary["versions"] == 2
    await call(client, "laya_save_schema", name="ops/alerts", questions={"bug": QUESTIONS["bug"]})
    assert len((await call(client, "laya_list_schemas", all_teams=True))["schemas"]) == 2
    assert (await call(client, "laya_list_schemas", team="ops"))["schemas"][0]["ref"].startswith("ops/alerts")


async def test_save_schema_defaults_and_errors(client, env):
    info = (await call(client, "laya_save_schema", name="plain", questions=QUESTIONS))["schema_info"]
    assert info["targets"]["min_accuracy"] == env.settings.default_target_accuracy
    assert info["targets"]["min_examples"] == env.settings.min_eval_examples

    with pytest.raises(ToolError, match="invalid questions"):
        await call(client, "laya_save_schema", name="broken",
                   questions={"c": {"type": "choice", "instructions": "Pick one"}})
    with pytest.raises(ToolError, match="invalid schema name"):
        await call(client, "laya_save_schema", name="Bad Name!", questions=QUESTIONS)
    with pytest.raises(ToolError, match="no version"):
        await call(client, "laya_save_schema", name="triage@2", questions=QUESTIONS)
    with pytest.raises(ToolError, match="unknown question ids: nope"):
        await call(client, "laya_save_schema", name="t", questions=QUESTIONS,
                   targets={"per_question": {"nope": 0.8}})
    with pytest.raises(ToolError, match="not found"):
        await call(client, "laya_get_schema", schema="dev/missing")
    with pytest.raises(ToolError, match="report 99 not found"):
        await call(client, "laya_get_report", report_id=99)


# --------------------------------------------------------------------------- datasets
async def test_save_and_list_datasets(client):
    first = await call(client, "laya_save_dataset", name="issues", examples=examples(8))
    assert first["dataset"]["count"] == 8 and first["added"] == 8
    assert first["dataset"]["question_ids"] == ["area", "bug"]
    assert any("promotion needs at least 10" in w for w in first["warnings"])
    more = await call(client, "laya_save_dataset", name="issues", examples=examples(4))
    assert more["dataset"]["count"] == 12
    replaced = await call(client, "laya_save_dataset", name="issues", examples=examples(3), append=False)
    assert replaced["dataset"]["count"] == 3
    listed = await call(client, "laya_list_datasets")
    assert [d["name"] for d in listed["datasets"]] == ["issues"]

    with pytest.raises(ToolError, match="chunks of 50"):
        await call(client, "laya_save_dataset", name="big", examples=examples(51))
    with pytest.raises(ToolError, match=r"example 1: `expected` is empty"):
        await call(client, "laya_save_dataset", name="bad",
                   examples=[{"state": "a", "expected": {"bug": True}}, {"state": "b", "expected": {}}])
    with pytest.raises(ToolError, match="empty question id"):
        await call(client, "laya_save_dataset", name="bad", examples=[{"state": "a", "expected": {"": True}}])
    with pytest.raises(ToolError, match="`state` is empty"):
        await call(client, "laya_save_dataset", name="bad", examples=[{"state": "  ", "expected": {"bug": True}}])


# --------------------------------------------------------------------------- evaluate -> calibrate -> promote
async def test_full_workflow_evaluate_and_promote(client, env):
    await save_schema(client)
    await call(client, "laya_save_dataset", name="issues", examples=examples(32))

    with pytest.raises(ToolError, match="has not been evaluated"):
        await call(client, "laya_promote_schema", schema="dev/triage")

    started, report_id = await evaluate(client)
    assert started["examples"] == 32 and "laya_job_status" in started["next_step"]
    job_meta = env.jobs.meta(started["job"]["job_id"])
    assert job_meta["schema_ref"] == "dev/triage@1" and job_meta["dataset"] == "dev/issues"

    report = await call(client, "laya_get_report", report_id=report_id)
    assert report["schema_ref"] == "dev/triage@1" and report["n_examples"] == 32
    assert report["per_question"]["area"]["accuracy"] == 1.0
    assert report["per_question"]["bug"]["accuracy"] == 1.0
    assert report["passes_targets"] and report["model_counts"] == {"english": 32}

    schema = await call(client, "laya_get_schema", schema="dev/triage")
    assert schema["status"] == "evaluated" and schema["latest_report_id"] == report_id
    assert set(schema["thresholds"]) == {"area", "bug"}

    promoted = await call(client, "laya_promote_schema", schema="dev/triage")
    assert promoted["schema_info"]["status"] == "trusted" and promoted["report_id"] == report_id
    again = await call(client, "laya_promote_schema", schema="dev/triage")
    assert "already trusted" in again["message"]

    # a clean re-evaluation keeps it trusted, a failing one demotes it
    await evaluate(client)
    assert (await call(client, "laya_get_schema", schema="dev/triage"))["status"] == "trusted"
    await call(client, "laya_save_dataset", name="noisy", examples=examples(32, flip_every=4))
    _, noisy_id = await evaluate(client, dataset="dev/noisy")
    assert not (await call(client, "laya_get_report", report_id=noisy_id))["passes_targets"]
    assert (await call(client, "laya_get_schema", schema="dev/triage"))["status"] == "evaluated"


async def test_promotion_refused_with_reasons(client):
    await save_schema(client)
    await call(client, "laya_save_dataset", name="noisy", examples=examples(40, flip_every=4))
    await evaluate(client, dataset="dev/noisy")
    with pytest.raises(ToolError) as err:
        await call(client, "laya_promote_schema", schema="dev/triage")
    msg = str(err.value)
    assert "cannot promote dev/triage@1" in msg
    assert "- bug: accuracy 75.0%" in msg and "90% target" in msg
    assert "area" not in msg.split("Next:")[0].replace("dev/triage", "")   # area meets its target
    assert "laya_calibrate" in msg and "laya_evaluate" in msg


async def test_promotion_refused_for_too_few_examples(client):
    await save_schema(client, targets={"min_accuracy": 0.9, "min_examples": 40})
    await call(client, "laya_save_dataset", name="issues", examples=examples(32))
    _, report_id = await evaluate(client)
    report = await call(client, "laya_get_report", report_id=report_id)
    assert not report["passes_targets"]
    with pytest.raises(ToolError, match="only 32 evaluated examples; need at least 40"):
        await call(client, "laya_promote_schema", schema="dev/triage")


async def test_evaluate_errors(client):
    await save_schema(client)
    with pytest.raises(ToolError, match="dataset dev/nope not found"):
        await call(client, "laya_evaluate", schema="dev/triage", dataset="nope")
    await call(client, "laya_save_dataset", name="other", examples=[{"state": "x", "expected": {"topic": "a"}}])
    with pytest.raises(ToolError, match="must be the schema's question ids"):
        await call(client, "laya_evaluate", schema="dev/triage", dataset="other")
    await call(client, "laya_save_dataset", name="few",
               examples=[{"state": "the database bug", "expected": {"area": "sideways", "bug": True}}])
    started = await call(client, "laya_evaluate", schema="dev/triage", dataset="few")
    warnings = "\n".join(started["warnings"])
    assert "area: 1 expected value(s) are not valid answers" in warnings
    assert "cannot pass promotion below 10" in warnings


async def test_calibrate(client, env):
    await save_schema(client)
    with pytest.raises(ToolError, match="no evaluation yet"):
        await call(client, "laya_calibrate", schema="dev/triage")
    await call(client, "laya_save_dataset", name="noisy", examples=examples(40, flip_every=4))
    started, report_id = await evaluate(client, dataset="dev/noisy")

    cal = await call(client, "laya_calibrate", schema="dev/triage")
    assert cal["schema_ref"] == "dev/triage@1"
    assert cal["n_fit"] + cal["n_holdout"] == 40 and cal["n_holdout"] == 12
    assert cal["temperatures"]["bug"] > 1.0          # over-confident on noisy labels -> softened
    assert cal["temperatures"]["area"] == 1.0        # always right -> degenerate, left alone
    assert set(cal["ece_before"]) == set(cal["ece_after"]) == {"area", "bug"}
    assert cal["report_id"] not in (None, report_id)

    schema = await call(client, "laya_get_schema", schema="dev/triage")
    assert schema["temperatures"] == cal["temperatures"]
    assert schema["latest_report_id"] == cal["report_id"]
    new_report = await call(client, "laya_get_report", report_id=cal["report_id"])
    assert new_report["calibrated"] and new_report["job_id"] == started["job"]["job_id"]
    assert new_report["per_question"]["bug"]["temperature"] == cal["temperatures"]["bug"]

    # explicit job id works too, and a batch job is refused
    again = await call(client, "laya_calibrate", schema="dev/triage", job_id=started["job"]["job_id"])
    assert again["temperatures"] == cal["temperatures"]
    batch = env.jobs.submit(team="dev", kind="batch", items=["x"], questions=QUESTIONS)
    with pytest.raises(ToolError, match="needs a laya_evaluate job"):
        await call(client, "laya_calibrate", schema="dev/triage", job_id=batch.job_id)
    with pytest.raises(ToolError, match="not found"):
        await call(client, "laya_calibrate", schema="dev/triage", job_id="nope")


async def test_calibrate_refuses_replaced_dataset(client):
    await save_schema(client)
    await call(client, "laya_save_dataset", name="issues", examples=examples(32))
    await evaluate(client)
    replacement = [{"state": f"other {i}", "expected": {"bug": True}} for i in range(32)]
    await call(client, "laya_save_dataset", name="issues", examples=replacement, append=False)
    with pytest.raises(ToolError, match="replaced"):
        await call(client, "laya_calibrate", schema="dev/triage")


async def test_calibrate_needs_enough_examples(client):
    await save_schema(client)
    await call(client, "laya_save_dataset", name="small", examples=examples(12))
    await evaluate(client, dataset="dev/small")
    with pytest.raises(ToolError, match="calibration needs at least 20"):
        await call(client, "laya_calibrate", schema="dev/triage")


# --------------------------------------------------------------------------- compare models
async def test_compare_models_agree(client):
    res = await call(client, "laya_compare_models", state="the database crashes with a bug", questions=QUESTIONS)
    assert set(res["models"]) == {"english", "multilingual", "typed-decisions"}
    assert res["agree"] and res["disagreements"] == [] and res["rows"] == 6
    assert res["models"]["english"]["answers"]["area"]["value"] == "backend"


async def test_compare_models_disagreement_and_schema(client, env):
    class Split(FakeEngine):
        async def predict_all_models(self, state, questions):
            out = await super().predict_all_models(state, questions)
            ans = out["typed-decisions"]["answers"]["area"]
            ans["probabilities"] = {"backend": 0.3, "frontend": 0.7}
            ans["choice"] = "frontend"
            return out

    env.engine = Split(env.settings)
    await save_schema(client)
    res = await call(client, "laya_compare_models", state="the database crashes with a bug", schema="dev/triage")
    assert not res["agree"]
    [d] = res["disagreements"]
    assert d["question_id"] == "area"
    assert d["values"] == {"english": "backend", "multilingual": "backend", "typed-decisions": "frontend"}
    assert d["confidences"]["typed-decisions"] == 0.7

    with pytest.raises(ToolError, match="exactly one"):
        await call(client, "laya_compare_models", state="x")
    many = {f"q{i}": {"type": "noul", "instructions": f"Is it {i}"} for i in range(14)}
    with pytest.raises(ToolError, match="3 checkpoints"):
        await call(client, "laya_compare_models", state="x", questions=many)


# --------------------------------------------------------------------------- sampling
async def test_sample_for_labeling_from_evaluate_job(client):
    await save_schema(client)
    await call(client, "laya_save_dataset", name="issues", examples=examples(32))
    started, _ = await evaluate(client)
    job_id = started["job"]["job_id"]

    unc = await call(client, "laya_sample_for_labeling", job_id=job_id, n=6, strategy="uncertain")
    assert unc["total_items"] == 32 and len(unc["items"]) == 6
    confs = [it["min_confidence"] for it in unc["items"]]
    assert confs == sorted(confs)
    first = unc["items"][0]
    assert first["reason"] == "uncertain"
    assert first["state"] == f"ticket {first['index']}: {TEMPLATES[first['index'] % 4][0]}"
    assert set(first["suggested_expected"]) == {"area", "bug"}
    assert "laya_save_dataset" in unc["next_step"]

    mixed = await call(client, "laya_sample_for_labeling", job_id=job_id, n=6)
    reasons = [it["reason"] for it in mixed["items"]]
    assert reasons.count("uncertain") == 3 and reasons.count("random") == 3
    assert len({it["index"] for it in mixed["items"]}) == 6

    r1 = await call(client, "laya_sample_for_labeling", job_id=job_id, n=5, strategy="random", seed=1)
    r2 = await call(client, "laya_sample_for_labeling", job_id=job_id, n=5, strategy="random", seed=1)
    assert [i["index"] for i in r1["items"]] == [i["index"] for i in r2["items"]]

    capped = await call(client, "laya_sample_for_labeling", job_id=job_id, n=200, strategy="uncertain")
    assert len(capped["items"]) == 32


async def test_sample_for_labeling_from_batch_job(client, env):
    batch = env.jobs.submit(team="dev", kind="batch", items=[t for t, _, _ in TEMPLATES] * 3, questions=QUESTIONS)
    res = await call(client, "laya_sample_for_labeling", job_id=batch.job_id, n=4, strategy="uncertain")
    assert len(res["items"]) == 4
    if hasattr(env.jobs, "states"):  # real JobService returns the submitted states
        assert all(it["state"] in [t for t, _, _ in TEMPLATES] for it in res["items"])
    else:
        assert all(it["state"] is None for it in res["items"])
        assert "by index" in res["next_step"]

    class StatefulJobs(FakeJobs):
        def states(self, job_id):
            return dict(enumerate(self._job(job_id)["items"]))

    env.jobs = StatefulJobs(env.repo, env.engine)
    batch = env.jobs.submit(team="dev", kind="batch", items=["alpha bug", "beta"], questions=QUESTIONS)
    res = await call(client, "laya_sample_for_labeling", job_id=batch.job_id, n=2, strategy="uncertain")
    assert {it["state"] for it in res["items"]} == {"alpha bug", "beta"}

    with pytest.raises(ToolError, match="not found"):
        await call(client, "laya_sample_for_labeling", job_id="missing")
