# Using laya-mcp from LangChain / LangGraph

laya-mcp is a plain MCP server — nothing about it is Claude-specific. Any
LangChain or LangGraph agent can use it via
[`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters),
which turns an MCP server's tools into ordinary LangChain `BaseTool`s.

> **Note on direction.** `langchain-mcp-adapters` is a *client* library: it
> lets a LangChain/LangGraph agent **consume** an MCP server. It is not a
> framework for building one — this server itself is built with FastMCP 4
> (see the project README). `langgraph` and a chat model integration (e.g.
> `langchain-anthropic`) are dependencies of *your* agent project, not of
> laya-mcp — install them there, not here.

## Minimal example

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "laya": {
            "url": "http://10.10.29.81:8765/mcp",
            "transport": "streamable_http",
            "headers": {"X-Laya-Team": "dev"},
        }
    }
)

tools = await client.get_tools()   # one LangChain tool per laya_* MCP tool
```

`tools` is a plain `list[BaseTool]` — pass it to any LangChain agent
constructor or LangGraph node the way you would your own tools. Each call
still costs real CPU time on the laya-mcp host (see
[dev-playbook.md](dev-playbook.md#cost-model-and-writing-concise-states)), so
the same guidance applies regardless of which agent framework calls it:
batch classify calls, keep states concise, prefer `laya_classify_batch` for
volume.

## Wiring it into a LangGraph agent

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient(
    {
        "laya": {
            "url": "http://10.10.29.81:8765/mcp",
            "transport": "streamable_http",
            "headers": {"X-Laya-Team": "support"},
        }
    }
)
tools = await client.get_tools()

agent = create_react_agent(model="anthropic:claude-sonnet-4-5", tools=tools)

result = await agent.ainvoke({
    "messages": [
        {"role": "user", "content": "Triage this ticket using dev/issue-triage: "
                                     "'Login button does nothing on Safari 17.'"}
    ]
})
print(result["messages"][-1].content)
```

`model=` and its credentials are your agent's own concern (this example uses
Claude, but any LangChain-supported chat model works identically — laya-mcp
doesn't care what called it). Requires `langgraph` in your project's own
dependencies; it is intentionally not a laya-mcp dependency.

## Multiple servers / calling a specific tool directly

`MultiServerMCPClient` can hold several servers at once (e.g. laya alongside
your own internal tools) — `get_tools(server_name="laya")` restricts to just
this one. To call a single tool directly instead of going through an agent's
tool-calling loop (e.g. for a deterministic pipeline step rather than an
autonomous agent):

```python
async with client.session("laya") as session:
    result = await session.call_tool(
        "laya_apply_preset",
        {"preset": "triage", "state": {"message": "My payment failed twice"}},
    )
```

## Keep the client in its own environment

`langchain-mcp-adapters` (0.3.1 and 0.3.2, the latest as of 2026-09) is built on the
MCP Python SDK 1.x: 0.3.1 fails to import next to `mcp` 2.x
(`cannot import name 'RequestContext' from 'mcp.shared.context'`), and 0.3.2 pins
`mcp<2`. The server needs `mcp` 2.x (FastMCP 4). The two never share an environment:
the client only talks HTTP, and FastMCP 4 negotiates the older protocol revision with
1.x clients.

```bash
python -m venv venv-clients
venv-clients\Scripts\python -m pip install "langchain-mcp-adapters==0.3.2"
venv-clients\Scripts\python scripts\langchain_smoke.py --url http://10.10.29.81:8765/mcp --team dev
```

`scripts/langchain_smoke.py` discovers the tools and calls `laya_status` and
`laya_classify`; exit code 0 means a LangChain agent can use the server. Never install
`langchain-mcp-adapters` into the server's venv: it downgrades `mcp` and breaks FastMCP.
