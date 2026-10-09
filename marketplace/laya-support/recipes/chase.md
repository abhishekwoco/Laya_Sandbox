# Recipe: stale-issue chaser

Find support issues that are still open and have gone quiet, and draft polite
in-thread nudges. Drafts only — the user approves every message before it is sent.

## 1. Collect

- Channel: named in the user's message, else **#errors-bugs** (resolve with
  `slack_search_channels`).
- Prefer today's run file `~/.laya-support/runs/<YYYY-MM-DD>.json` (or yesterday's)
  as the issue list — it already has condensed items and resolved/unresolved
  answers. If no run file exists, collect and classify exactly as in the triage
  recipe (steps 1–3), questions `resolved` and `category` only.

## 2. Select what to chase

An issue qualifies when ALL of these hold:

- `resolved` = false (if that answer was `needs_review`, read the thread and decide
  yourself before chasing — never nudge a thread that actually got resolved),
- no activity for 24 hours or more,
- the ball is with the dev/product side (the last substantive message is the
  reporter asking or waiting — NOT the dev asking the reporter for information;
  if the reporter owes the reply, skip it or note it separately as "waiting on
  reporter"),
- the newest thread message is not already a nudge from us
  (contains "gentle reminder" or "following up") less than 48 hours old.

## 3. Draft the nudges

One short in-thread reply per issue, addressed to whoever was last asked to act
(the person @mentioned or the dev who said they'd check). Tone: brief, factual,
no blame:

> Gentle reminder on this one — it's been <N> days with no update and
> <reporter> is waiting. Any progress?

Vary the wording naturally; mention a deadline only if the thread itself states one.

## 4. Approve and send

Show ALL drafts in one list (issue → draft text). Let the user approve all, some,
or edit. Post only the approved ones with `slack_send_message` using the thread's
`thread_ts`. Report what was posted and what was skipped.
