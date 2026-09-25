#!/usr/bin/env python
"""Download the Laya checkpoints into the local Hugging Face cache, so the server can run with
HF_HUB_OFFLINE=1.

It fetches exactly the files laya.Agent reads (rl_agent_config.json, model.safetensors,
tokenizer/*, encoder/*; see laya/agent.py) for each checkpoint in laya.router.DEFAULT_MODELS,
which are subfolders of one bundle repo. Nothing is built in memory, so downloading all three
needs no more RAM than a normal download. About 2 GB goes to disk for multilingual +
typed-decisions (english is ~1.7 GB on its own).

    venv\\Scripts\\python.exe scripts\\prefetch_models.py                       # all three
    venv\\Scripts\\python.exe scripts\\prefetch_models.py --models multilingual typed-decisions
    venv\\Scripts\\python.exe scripts\\prefetch_models.py --verify              # then load each offline

--verify loads each checkpoint one at a time with HF_HUB_OFFLINE=1 and runs a one-question
prediction, which proves the server will start without network access. Set HF_TOKEN if the repo
ever needs authentication; HF_HOME / HF_HUB_CACHE choose the cache location as usual.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

AGENT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")  # laya 0.3.7
TRUTHY = ("1", "true", "yes", "on")


def _checkpoint_size(target: Path) -> int:
    """Bytes of the files laya.Agent reads from one checkpoint directory."""
    total = sum((target / f).stat().st_size for f in ("rl_agent_config.json", "model.safetensors")
                if (target / f).exists())
    for d in ("tokenizer", "encoder"):
        if (target / d).is_dir():
            total += sum(p.stat().st_size for p in (target / d).rglob("*") if p.is_file())
    return total


def download(names: list[str]) -> int:
    from huggingface_hub import snapshot_download

    from laya.router import DEFAULT_MODELS

    failures = 0
    for name in names:
        repo, sub = DEFAULT_MODELS[name]
        prefix = f"{sub}/" if sub else ""
        t0 = time.perf_counter()
        print(f"[{name}] downloading {repo}{'/' + sub if sub else ''} ...", flush=True)
        try:
            path = Path(snapshot_download(
                repo,
                allow_patterns=[prefix + f for f in AGENT_FILES],
                token=os.environ.get("HF_TOKEN"),
            ))
        except Exception as e:
            failures += 1
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        target = path / sub if sub else path
        missing = [f for f in ("rl_agent_config.json", "model.safetensors") if not (target / f).exists()]
        if missing:
            failures += 1
            print(f"[{name}] FAILED: {missing} missing under {target}", file=sys.stderr, flush=True)
            continue
        size = _checkpoint_size(target)
        print(f"[{name}] ok: {target} ({size / 2**30:.2f} GB, {time.perf_counter() - t0:.0f}s)", flush=True)
    return failures


def verify(names: list[str]) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = True  # the module read the variable at import time
    except Exception:
        pass
    import laya
    from laya.router import DEFAULT_MODELS

    probe = {"ok": {"type": "noul", "instructions": "Is this a verification request?"}}
    failures = 0
    for name in names:
        repo, sub = DEFAULT_MODELS[name]
        t0 = time.perf_counter()
        try:
            agent = laya.load(repo, subfolder=sub)
            answer = agent.system_one("This is a verification request.", probe)["answers"]["ok"]
            print(f"[{name}] offline load + predict ok in {time.perf_counter() - t0:.1f}s "
                  f"(p_true={answer['noul']:.2f})", flush=True)
            del agent
        except Exception as e:
            failures += 1
            print(f"[{name}] offline verification FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        gc.collect()
    return failures


def main() -> int:
    from laya.router import DEFAULT_MODELS, normalise_name

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                    help="checkpoints to fetch (default: all of %s)" % ", ".join(DEFAULT_MODELS))
    ap.add_argument("--verify", action="store_true", help="afterwards load each checkpoint offline and predict once")
    ap.add_argument("--verify-only", action="store_true", help="skip downloading; only verify the cache")
    args = ap.parse_args()
    names = [normalise_name(m) for m in args.models]

    failures = 0
    if not args.verify_only:
        if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in TRUTHY:
            print("HF_HUB_OFFLINE is set, so nothing can be downloaded. Unset it for this command "
                  "(PowerShell: Remove-Item Env:HF_HUB_OFFLINE) or use --verify-only.", file=sys.stderr)
            return 2
        failures += download(names)
    if args.verify or args.verify_only:
        failures += verify(names)
    if failures:
        print(f"{failures} step(s) failed", file=sys.stderr)
        return 1
    print("done: run the server with HF_HUB_OFFLINE=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
