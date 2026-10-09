---
name: triage
description: Run the WoCo support team's daily triage digest - read the support Slack channel and its threads, classify every issue with Laya (bug vs config, urgency, resolved, escalation, channel rules), pair duplicates, and draft one digest message for the user's approval. Use when the user invokes /triage or asks to run the daily support triage. Arguments may name a channel (e.g. "/triage #support-helpdesk").
---

# Daily support triage digest

You are running the WoCo support team's daily triage. Follow these steps exactly.
Everything that posts to Slack is a DRAFT the user approves first — never send
without showing the final text and getting a yes in chat.

## 0. Channel and window

- Channel: if the arguments name a channel (e.g. `/triage #support-helpdesk`),
  use that. Otherwise default to **#errors-bugs**. Resolve the id with
  `slack_search_channels`.
- Dedup guard: read the latest ~30 channel messages first. If one already contains
  `Laya triage — <today's date>` (format: `09 Oct 2026`), STOP and tell the user
  today's digest already exists (link it). Do not build a second one.
- Window: from the timestamp of the previous `Laya triage —` digest message if you
  can find one in the last 7 days; otherwise the last 24 hours.

## 1. Collect

- `slack_read_channel` with `oldest` = window start.
- For every message that has a thread, `slack_read_thread` to get the replies —
  resolution signals usually live in the replies.
- Group messages into DISTINCT ISSUES (an inline back-and-forth without a thread is
  still one issue). Ignore pure chatter (thanks, 👍, done confirmations that belong
  to an issue you already grouped).
- Messages whose only content is an image/screenshot with no describing text cannot
  be classified — collect them into an "image-only" list with permalink and author;
  they go in the digest for human eyes, not through Laya.

## 2. Condense

For each issue build a compact item (keep each under ~400 tokens):

```json
{"id": "<short-slug>",
 "issue": "<what is being reported, tenant/employee names kept>",
 "discussion": "<how the thread went: who replied what, in order, condensed>",
 "last_activity": "<ISO datetime of the newest message in the issue>",
 "permalink": "<link to the parent message>"}
```

## 3. Classify with Laya

First check `laya_list_schemas` for a schema named `ticket-intake` (team support):

- If it exists with status **trusted** → use `laya_decide` with it.
- Otherwise → `laya_classify` with these questions and `threshold: 0.8`:

```json
{
 "category":  {"type": "choice",
   "instructions": "What kind of item is `issue`, given the outcome in `discussion`?",
   "criteria": {
     "product_bug":    "a defect in WoCo's application code or templates that developers must fix with a code change",
     "config_or_user": "not a code defect: settings, permissions, data setup, outdated app version, wrong credentials, or a routine ops request",
     "question":       "a question or information request, nothing is broken",
     "other":          "announcement, customization request, or anything that is not an issue report"}},
 "urgency":   {"type": "score",
   "instructions": "How urgent is the item in `issue` and `discussion`?",
   "criteria": ["cosmetic or no deadline", "affects someone's work but has a workaround",
                "blocks a core workflow or a payroll/compliance deadline is near",
                "many users blocked or data is wrong in production"]},
 "resolved":  {"type": "noul",
   "instructions": "Based on `discussion`, was the issue in `issue` actually resolved by the end of the conversation?",
   "criteria": {"true": "fix or request completed, works now, or reporter confirmed/stopped complaining after a working answer",
                "false": "still open: awaiting response, being worked on, unanswered, or reporter still blocked"}},
 "escalation": {"type": "noul",
   "instructions": "Does `discussion` show frustration, repeated chasing, or deadline/SLA risk that a team lead should see today?"},
 "valid_post": {"type": "noul",
   "instructions": "Is `issue` an error/bug report (allowed in this channel), as opposed to a customization request or logic discussion (not allowed)?"}
}
```

Budget: rows = items x 5 questions and synchronous calls allow 20 rows. With more
than 4 items, split into several `laya_classify` calls of at most 4 items each; with
more than 20 items use `laya_classify_batch` and poll `laya_job_status`.

Answers flagged `needs_review` (or `unverified` from a draft schema) are YOURS to
decide: read the item, decide yourself, and mark that answer with ✱ in the digest so
the team knows a human/agent call was applied over an unsure Laya answer.

## 4. Duplicates

Among items with the same `category` that are unresolved, run pairwise `noul`
checks ("Do `a` and `b` describe the same underlying issue?") with both condensed
items in the state. Only likely pairs — cap at 10 pairs. Confident "yes" pairs go
in the digest's duplicates section.

## 5. Compose the digest

```
:clipboard: Laya triage — <DD Mon YYYY> (#<channel>)

:fire: Escalations (<n>)
• <item> — <why it needs a lead today> <permalink>

:new: New issues (<n>)
1. [BUG] <one-line issue> — urgency <level>, <open/resolved>, last reply <who/when> <permalink>
2. [CONFIG✱] ...

:hourglass: Still open, waiting on us (<n>)
• <item> — silent since <when>, last asked by <who> <permalink>

:repeat: Possible duplicates
• <item A> ↔ <item B>

:no_entry_sign: Misplaced posts (channel rules)
• <item> — looks like a customization request

:eyes: Image-only, needs human eyes
• <author>: <permalink>

Corrections? Reply in this thread like "2 = config" or "3 resolved" — replies train Laya.
```

Omit empty sections. ✱ = Laya was unsure, judgement applied. Keep the digest under
~40 lines; link instead of quoting.

## 6. Save the run file (required — the /labels sweep depends on it)

Write the full run to `~/.laya-support/runs/<YYYY-MM-DD>.json`: the channel, window,
every condensed item, Laya's raw answers (with confidences/status), and your final
applied answers. Create the directory if needed.

## 7. Post

Show the digest to the user and ask to send. On yes, post it to the channel with
`slack_send_message`. Never post without the explicit yes.
