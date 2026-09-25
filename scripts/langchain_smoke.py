"""Cross-agent smoke test: a LangChain client (langchain-mcp-adapters, mcp 1.x SDK) uses laya-mcp.

Runs from its own venv because langchain-mcp-adapters (<=0.3.2) requires mcp<2 while the server
needs mcp 2.x:

    python -m venv venv-clients
    venv-clients\\Scripts\\python -m pip install "langchain-mcp-adapters==0.3.2"
    venv-clients\\Scripts\\python scripts\\langchain_smoke.py --url http://localhost:8765/mcp

It also proves that clients built on the older MCP SDK can talk to the FastMCP 4 server.
Exit code 0 = tools discovered and laya_status + laya_classify returned results.
"""
import argparse
import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

QUESTIONS = {
    "area": {
        "type": "choice",
        "instructions": "Which part of the system is the bug report about?",
        "criteria": {"api": "HTTP endpoints, request handling", "database": "queries, migrations, storage", "ui": "screens, layout"},
    },
    "has_repro": {"type": "noul", "instructions": "Does the report include steps to reproduce?"},
}


def _payload(result):
    """Tool results come back as content blocks (or a string); return the parsed JSON body."""
    if isinstance(result, tuple):          # (content, artifact) with response_format content_and_artifact
        result = result[0]
    if isinstance(result, list):
        result = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in result)
    return json.loads(result) if isinstance(result, str) else result


async def main(url: str, team: str) -> int:
    client = MultiServerMCPClient(
        {"laya": {"transport": "streamable_http", "url": url, "headers": {"X-Laya-Team": team}}}
    )
    tools = {t.name: t for t in await client.get_tools()}
    print(f"discovered {len(tools)} tools")
    for required in ("laya_status", "laya_classify", "laya_decide", "laya_evaluate"):
        if required not in tools:
            print(f"FAIL: {required} not exposed", file=sys.stderr)
            return 1

    status = _payload(await tools["laya_status"].ainvoke({}))
    engine = status.get("engine", status)
    print("status:", json.dumps({"loaded": engine.get("loaded"), "device": engine.get("device"), "team": status.get("team")}))

    out = _payload(
        await tools["laya_classify"].ainvoke(
            {"questions": QUESTIONS, "state": {"title": "500 error on POST /orders after DB migration", "body": "Steps: 1. run migration 2. POST /orders"}}
        )
    )
    answers = out["results"][0]["answers"]
    print("classify:", {q: (a["value"], a["confidence"]) for q, a in answers.items()}, f"{out['latency_ms']} ms")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765/mcp")
    ap.add_argument("--team", default="dev")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.url, args.team)))
