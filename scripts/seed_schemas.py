"""Seed the schema library with the example schemas under examples/schemas/.

Loads every examples/schemas/<team>/<name>.json file (see
examples/schemas/dev/*.json for the four dev-team presets) and saves each one
through a *running* laya-mcp server via the MCP `laya_save_schema` tool, using
an ordinary MCP client -- this script has no special access, it talks to the
server exactly the way any other MCP client would.

Each JSON file must have the shape:
    {"name": str, "description": str, "questions": {...}, "targets": {...}}
The team namespace is taken from the file's parent directory name (so
examples/schemas/dev/issue-triage.json becomes dev/issue-triage).

The server must already be up (see docs/operations.md) -- this script is a
client, it does not start or configure anything. It's safe to re-run:
laya_save_schema always creates a new draft version, it never overwrites or
deletes history, so re-seeding just adds versions to promote or discard later.

Usage:
    venv\\Scripts\\python.exe scripts\\seed_schemas.py
    venv\\Scripts\\python.exe scripts\\seed_schemas.py --url http://10.10.29.81:8765/mcp
    venv\\Scripts\\python.exe scripts\\seed_schemas.py --dir examples\\schemas\\dev
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client

DEFAULT_URL = "http://localhost:8765/mcp"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "examples" / "schemas"


def find_schema_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.json"))


def load_schema(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("name", "questions"):
        if key not in data:
            raise ValueError(f"missing required key {key!r}")
    return data


async def seed(url: str, root: Path) -> int:
    files = find_schema_files(root)
    if not files:
        print(f"No schema JSON files found under {root}", file=sys.stderr)
        return 1

    failures = 0
    async with Client(url) as client:
        for path in files:
            team = path.parent.name
            try:
                data = load_schema(path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(f"SKIP {path}: {exc}", file=sys.stderr)
                failures += 1
                continue

            ref = f"{team}/{data['name']}"
            arguments = {
                "team": team,
                "name": data["name"],
                "questions": data["questions"],
                "description": data.get("description", ""),
                "targets": data.get("targets", {}),
            }
            try:
                result = await client.call_tool("laya_save_schema", arguments)
            except Exception as exc:  # noqa: BLE001 - report and keep seeding the rest
                print(f"FAIL {ref}: {exc}", file=sys.stderr)
                failures += 1
                continue

            payload = getattr(result, "data", result)
            print(f"OK   {ref}  ->  {payload}")

    if failures:
        print(f"\n{failures} of {len(files)} schema(s) failed to seed.", file=sys.stderr)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"MCP server URL (default: {DEFAULT_URL})")
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help=f"Directory of schema JSON files, searched recursively (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    raise SystemExit(asyncio.run(seed(args.url, root)))


if __name__ == "__main__":
    main()
