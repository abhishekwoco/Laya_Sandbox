"""Question validation and request-size helpers used by InferenceEngine.

Everything here is pure (no inference, no locks) and raises plain `ValueError`; the engine turns
those into `InvalidQuestions` / `BudgetExceeded` (engine/runtime.py) so this module does not
import the runtime.

Structural checks reuse laya's own `Agent._check_question` so error messages match what laya
would raise at inference time. When a tokenizer and checkpoint config are available, the option
layout of every question is computed the way `Agent.system_one` builds it
(laya/agent.py:345-351 -> laya.common.build_sequence + render_options), which is how we detect
questions whose options cannot all get a marker, truncated label descriptions and truncated
instructions before any forward pass.
"""
from __future__ import annotations

import difflib
import functools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_CHOICE_LABELS_WARN = 12
MAX_INSTRUCTIONS_CHARS_WARN = 300
OPTION_TOKEN_CAP = 48            # build_sequence keeps at most 48 tokens per option text
MIN_OPTION_BUDGET = 16           # build_sequence starts cutting options below this
NEAR_BUDGET_FRACTION = 0.8       # warn when options use more than this share of their budget
MIN_STATE_ROOM_WARN = 128        # warn when fewer tokens than this remain for the state
NEAR_DUPLICATE_RATIO = 0.85
_QTYPES = ("choice", "noul", "score")


# --------------------------------------------------------------------------- laya reuse
@functools.cache
def _laya_agent_cls() -> Any | None:
    try:
        from laya.agent import Agent  # imports torch; only loaded on first validation
    except Exception:  # pragma: no cover - laya is a hard dependency
        return None
    return Agent


def _check_question_fallback(qid: str, qdef: Any) -> None:  # pragma: no cover - mirrors laya 0.3.7
    if not isinstance(qdef, dict):
        raise ValueError("question %r: definition must be a dict, got %s" % (qid, type(qdef).__name__))
    t = qdef.get("type")
    if t not in _QTYPES:
        raise ValueError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(_QTYPES)))
    if "instructions" not in qdef:
        raise ValueError("question %r: no 'instructions'; add the text the model should answer" % (qid,))
    crit = qdef.get("criteria")
    if t == "choice":
        if not isinstance(crit, (dict, list)):
            raise ValueError("question %r: a choice question takes 'criteria' as a dict of "
                             "label -> description, or a list of labels" % (qid,))
        if not crit:
            raise ValueError("question %r: a choice question needs at least one criterion" % (qid,))
    elif t == "score":
        if not isinstance(crit, list):
            raise ValueError("question %r: a score question takes 'criteria' as a list of level "
                             "descriptions, index 0 first" % (qid,))
        if not crit:
            raise ValueError("question %r: a score question needs at least one level" % (qid,))
    elif crit is not None and not isinstance(crit, dict):
        raise ValueError("question %r: a noul question takes 'criteria' as a dict with optional "
                         "'true'/'false' descriptions, or omits it" % (qid,))


def laya_check_question(qid: str, qdef: Any) -> None:
    """laya's own structural check (`Agent._check_question` is a staticmethod)."""
    agent = _laya_agent_cls()
    fn = getattr(agent, "_check_question", None) if agent is not None else None
    (fn or _check_question_fallback)(qid, qdef)


def to_internal(qdef: dict[str, Any]) -> dict[str, Any]:
    """laya's internal question form {"t", "ins", "crit"} (`Agent._to_internal`)."""
    agent = _laya_agent_cls()
    fn = getattr(agent, "_to_internal", None) if agent is not None else None
    if fn is not None:
        return fn(qdef)
    t, crit = qdef["type"], qdef.get("criteria")  # pragma: no cover - mirrors laya 0.3.7
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    return {"t": t, "ins": qdef["instructions"], "crit": crit}


def render_options(q_internal: dict[str, Any]) -> list[str]:
    from laya.common import render_options as _render

    return _render(q_internal)


def serialize_state(state: Any) -> str:
    """Exactly the text laya tokenizes for a state (laya.common.serialize_state)."""
    try:
        from laya.common import serialize_state as _ser
    except Exception:  # pragma: no cover
        import json

        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return _ser(state)


# --------------------------------------------------------------------------- sizes
def rows_for(n_items: int, questions: Mapping[str, Any] | Sequence[Any] | int) -> int:
    """Rows a request costs: one row per (item, question). Inference time scales with rows x
    tokens per row, and each state is one forward pass over all of its question rows."""
    n_q = questions if isinstance(questions, int) else len(questions)
    return max(0, int(n_items)) * max(0, int(n_q))


def state_chars(state: Any) -> int:
    return len(serialize_state(state))


def oversized_states(states: Sequence[Any], max_chars: int) -> list[tuple[int, int]]:
    """(index, serialized length) for every state longer than `max_chars`."""
    out = []
    for i, s in enumerate(states):
        n = state_chars(s)
        if n > max_chars:
            out.append((i, n))
    return out


# --------------------------------------------------------------------------- structure
def check_structure(questions: Any) -> None:
    """Raise ValueError (laya wording where laya has one) for a question set laya cannot answer."""
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError(
            "no questions: send at least one question as {id: {type, instructions, criteria}}"
        )
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("question %r: question ids must be non-empty strings" % (qid,))
        laya_check_question(qid, q)
        ins = q["instructions"]
        if not isinstance(ins, str) or not ins.strip():
            raise ValueError("question %r: 'instructions' must be non-empty text, e.g. "
                             "'Which team should handle `message`?'" % (qid,))
        crit = q.get("criteria")
        if q["type"] == "choice":
            labels = list(crit) if isinstance(crit, list) else list(crit.keys())
            for lab in labels:
                if not isinstance(lab, str) or not lab.strip():
                    raise ValueError("question %r: choice labels must be non-empty strings, got %r"
                                     % (qid, lab))
            if isinstance(crit, list):
                seen: set[str] = set()
                for lab in labels:
                    if lab in seen:
                        raise ValueError("question %r: label %r appears twice; labels must be unique"
                                         % (qid, lab))
                    seen.add(lab)


# --------------------------------------------------------------------------- token layout
@dataclass
class OptionLayout:
    """How build_sequence lays out one question's head (instructions + option markers)."""

    n_options: int
    markers: int                    # option markers that fit inside max_len
    option_tokens: int              # tokens the options take after any truncation
    option_budget: int              # head_max_len - MIN_OPTION_BUDGET: what options may use untruncated
    per_option_cap: int | None      # set when build_sequence had to cut every option to this length
    head_tokens: int                # instruction tokens before truncation
    head_kept: int                  # instruction tokens kept
    state_room: int                 # tokens left for the state
    long_options: list[int] = field(default_factory=list)   # indexes of options over 48 tokens
    raw_option_tokens: list[int] = field(default_factory=list)


def _encode(tok: Any, text: str) -> list[int]:
    mask = getattr(tok, "mask_token", None)
    if isinstance(mask, str) and mask:
        text = text.replace(mask, " ")
    return list(tok(text, add_special_tokens=False)["input_ids"])


def option_layout(tok: Any, q_internal: dict[str, Any], max_len: int = 512, head_max_len: int = 192) -> OptionLayout:
    """Mirror of laya.common.build_sequence for the question head, without a state.

    Works with any tokenizer exposing `tok(text, add_special_tokens=False)["input_ids"]`, so it
    also runs with test fakes that have no special tokens."""
    opts = render_options(q_internal)
    head_full = _encode(tok, "%s question: %s" % (q_internal["t"], q_internal["ins"]))
    raw = [len(_encode(tok, " " + o)) for o in opts]
    lens = [1 + min(OPTION_TOKEN_CAP, n) for n in raw]          # [MASK] + option tokens[:48]
    opt_budget = head_max_len - sum(lens)
    per = None
    if opt_budget < MIN_OPTION_BUDGET:
        per = max(4, (head_max_len - MIN_OPTION_BUDGET) // max(1, len(lens)))
        lens = [min(n, per) for n in lens]
        opt_budget = head_max_len - sum(lens)
    head_kept = min(len(head_full), max(8, opt_budget))
    pos = 1 + head_kept + 1                                       # [CLS] head [SEP]
    markers = []
    for n in lens:
        markers.append(pos)
        pos += n
    pos += 1                                                      # [SEP] after the options
    return OptionLayout(
        n_options=len(opts),
        markers=sum(1 for m in markers if m < max_len),
        option_tokens=sum(lens),
        option_budget=head_max_len - MIN_OPTION_BUDGET,
        per_option_cap=per,
        head_tokens=len(head_full),
        head_kept=head_kept,
        state_room=max(0, max_len - pos - 1),
        long_options=[i for i, n in enumerate(raw) if n > OPTION_TOKEN_CAP],
        raw_option_tokens=raw,
    )


def _laya_marker_count(tok: Any, q_internal: dict[str, Any], max_len: int, head_max_len: int) -> int | None:
    """Markers laya itself produces (agent.py:348), when the tokenizer is a real HF tokenizer."""
    if any(getattr(tok, a, None) is None for a in ("mask_token", "mask_token_id", "cls_token_id", "sep_token_id")):
        return None
    try:
        from laya.common import build_sequence
    except Exception:  # pragma: no cover
        return None
    _, markers = build_sequence(tok, "", q_internal, max_len, head_max_len)
    return len(markers)


# --------------------------------------------------------------------------- warnings
def _norm_text(s: Any) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if len(ta) >= 3 and len(tb) >= 3 and len(ta & tb) / len(ta | tb) >= 0.8:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= NEAR_DUPLICATE_RATIO


def _duplicate_pairs(named: list[tuple[str, str]]) -> list[tuple[str, str, bool]]:
    """(name_a, name_b, exact) for descriptions that are identical or nearly so."""
    normed = [(n, _norm_text(d)) for n, d in named if d not in (None, "")]
    out = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            (na, da), (nb, db) = normed[i], normed[j]
            if _similar(da, db):
                out.append((na, nb, da == db))
    return out


def structural_warnings(qid: str, q: dict[str, Any]) -> list[str]:
    """Advice that needs no tokenizer. `q` must already pass check_structure."""
    w: list[str] = []
    t, crit, ins = q["type"], q.get("criteria"), q["instructions"]
    if len(ins) > MAX_INSTRUCTIONS_CHARS_WARN:
        w.append("question %r: instructions are %d characters; keep them under %d and put detail in the "
                 "label descriptions, because long instructions eat the option budget and get truncated"
                 % (qid, len(ins), MAX_INSTRUCTIONS_CHARS_WARN))
    if t == "choice":
        items = [(lab, None) for lab in crit] if isinstance(crit, list) else list(crit.items())
        labels = [lab for lab, _ in items]
        if len(labels) > MAX_CHOICE_LABELS_WARN:
            w.append("question %r has %d labels; accuracy and calibration drop past %d options. Split it into "
                     "a coarse question plus a follow-up, or pre-filter the candidates" % (qid, len(labels), MAX_CHOICE_LABELS_WARN))
        if len(labels) == 1:
            w.append("question %r has a single label, so it always answers %r with probability 1.0; "
                     "use a noul question for yes/no" % (qid, labels[0]))
        bare = [lab for lab, d in items if d in (None, "")]
        if bare and len(bare) < len(items):
            w.append("question %r: labels %s have no description while others do; describe every label so "
                     "the options are comparable" % (qid, ", ".join(repr(b) for b in bare)))
        elif bare and len(labels) > 1:
            w.append("question %r: labels have no descriptions; bare labels work only when the names are "
                     "self-explanatory, a short description per label usually improves accuracy" % (qid,))
        lower: dict[str, str] = {}
        for lab in labels:
            key = _norm_text(lab)
            if key in lower:
                w.append("question %r: labels %r and %r differ only in case or punctuation; merge them"
                         % (qid, lower[key], lab))
            else:
                lower[key] = lab
        for a, b, exact in _duplicate_pairs([(lab, d) for lab, d in items]):
            w.append("question %r: labels %r and %r have %s descriptions, so the model cannot tell them apart; "
                     "rewrite them to state what distinguishes each" % (qid, a, b, "identical" if exact else "near-identical"))
    elif t == "score":
        if len(crit) == 1:
            w.append("question %r has a single level, so its score is always 0; add levels" % (qid,))
        empty = [i for i, d in enumerate(crit) if d is None or (isinstance(d, str) and not d.strip())]
        if empty:
            w.append("question %r: score levels %s have no description; describe every level, index 0 = lowest"
                     % (qid, ", ".join(str(i) for i in empty)))
        for a, b, exact in _duplicate_pairs([("level %d" % i, d) for i, d in enumerate(crit)]):
            w.append("question %r: %s and %s have %s descriptions; each level should describe a distinct degree"
                     % (qid, a, b, "identical" if exact else "near-identical"))
    else:  # noul
        if isinstance(crit, dict):
            extra = [k for k in crit if str(k).lower() not in ("true", "false")]
            if extra:
                w.append("question %r: noul criteria keys %s are ignored; only 'true' and 'false' are read"
                         % (qid, ", ".join(repr(k) for k in extra)))
    return w


def layout_findings(qid: str, q: dict[str, Any], tok: Any, cfg: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """(error, warnings) from the checkpoint's token budget for one question."""
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    qi = to_internal(q)
    lay = option_layout(tok, qi, max_len, head_max_len)
    markers = lay.markers
    real = _laya_marker_count(tok, qi, max_len, head_max_len)
    if real is not None:
        markers = real
    if markers != lay.n_options:
        return (
            "question %r options exceed head_max_len=%d: %d options but only %d fit inside the %d-token "
            "input. Remove labels or split the question into a coarse question plus a follow-up"
            % (qid, head_max_len, lay.n_options, markers, max_len),
            [],
        )
    w: list[str] = []
    if lay.per_option_cap is not None:
        w.append("question %r: the options need more than the %d-token option budget (head_max_len=%d), so "
                 "every option is cut to %d tokens and label descriptions are truncated. Shorten descriptions, "
                 "drop labels or split the question" % (qid, lay.option_budget, head_max_len, lay.per_option_cap))
    elif lay.option_tokens > NEAR_BUDGET_FRACTION * lay.option_budget:
        w.append("question %r: options use %d of the %d-token option budget; adding labels or longer "
                 "descriptions will truncate them" % (qid, lay.option_tokens, lay.option_budget))
    if lay.long_options:
        opts = render_options(qi)
        names = [opts[i].split(":", 1)[0] for i in lay.long_options]
        w.append("question %r: the description of %s is longer than %d tokens; only the first %d are read"
                 % (qid, ", ".join(repr(n) for n in names), OPTION_TOKEN_CAP, OPTION_TOKEN_CAP))
    if lay.head_kept < lay.head_tokens:
        w.append("question %r: instructions are cut from %d to %d tokens because the options fill the head; "
                 "shorten the instructions or the descriptions" % (qid, lay.head_tokens, lay.head_kept))
    if lay.state_room < MIN_STATE_ROOM_WARN:
        w.append("question %r: only %d tokens remain for the state after instructions and options (max_len=%d); "
                 "most of the content will be cut" % (qid, lay.state_room, max_len))
    return None, w


def validate_questions(
    questions: Any,
    *,
    tok: Any | None = None,
    cfg: Mapping[str, Any] | None = None,
    max_questions: int | None = None,
) -> list[str]:
    """Raise ValueError for an unanswerable question set; otherwise return warnings.

    With `tok` and `cfg` (a loaded checkpoint), also checks that every option gets a marker inside
    the checkpoint's input and warns about truncation."""
    check_structure(questions)
    warnings: list[str] = []
    if max_questions is not None and len(questions) > max_questions:
        warnings.append("%d questions exceed the per-call limit of %d; synchronous calls will be refused, "
                        "so split the set or use laya_classify_batch" % (len(questions), max_questions))
    for qid, q in questions.items():
        warnings.extend(structural_warnings(qid, q))
        if tok is not None and cfg is not None:
            err, w = layout_findings(qid, q, tok, cfg)
            if err:
                raise ValueError(err)
            warnings.extend(w)
    return warnings
