# CLAUDE.md snippet: laya-mcp

Paste the block below into your project's `CLAUDE.md` once the `laya` MCP
server is added (see [client-setup.md](client-setup.md)). Trim it to the
tools/schemas your project actually uses.

---

```markdown
## Using the laya MCP server

`laya` is a fast, calibrated classifier available as MCP tools
(`laya_classify`, `laya_decide`, `laya_apply_preset`, `laya_classify_batch`,
`laya_scan_untrusted`, and more — see D:\Laya_Sandbox\docs\tools.md).

Delegate to it for bulk/typed decisions instead of reasoning them out
yourself: issue/ticket triage, log classification, tagging, filtering
research sources by relevance, or a first-pass scan of untrusted fetched
text. Do NOT delegate open-ended reasoning, planning, or code-correctness
judgments — those stay with you.

Workflow:
1. Extract a concise `state` (just what's needed to answer the questions —
   not the whole document) and pick a saved schema if one exists for this
   kind of decision (`laya_list_schemas`), otherwise use `laya_classify` with
   ad hoc questions or `laya_apply_preset`.
2. Call `laya_decide` (schema-based) or `laya_classify`. For more than a
   handful of items, use `laya_classify_batch` and poll
   `laya_job_status`/`laya_job_results` instead of blocking on one big call.
3. Trust `status: "decided"` answers without re-checking them. For
   `status: "needs_review"`, reason about that specific question yourself —
   that's the point of the confidence gate. For `status: "unverified"`
   (schema not yet evaluated), treat the answer as a rough suggestion and
   verify anything you're about to act on.

Every item costs real CPU time (roughly 0.5-1.6 s per question (longer input costs more) on the host),
so don't fire one call per item — batch questions together, and use
`laya_classify_batch` for anything beyond a handful of items.
```
