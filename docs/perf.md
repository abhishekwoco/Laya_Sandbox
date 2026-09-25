# Laya inference performance on the server host

Phase 0 performance spike, measured 2026-09-25 with `scripts/bench.py` (english checkpoint, CPU,
`HF_HUB_OFFLINE=1`). Host: Ryzen 7 5800H (8 cores / 16 threads), 16 GB RAM, torch 2.14 CPU-only,
laya 0.3.7, Python 3.14. MySQL, Postgres, Ollama, httpd and other apps ran during the benchmark;
see [Variance](#variance) before comparing single cells.

## Recommended settings

| Setting (env) | Recommended | Current default | Why |
|---|---|---|---|
| `LAYA_MCP_THREADS` | **6** | 6 | Interleaved runs: 8 threads is only 7-8% faster than 6 (see [Threads](#threads)). 8 intra-op threads occupy every physical core and are the most exposed to stalls when the databases or Ollama get busy. Use 8 only on a dedicated host. |
| `LAYA_MCP_QUANTIZE` | **false** | false | int8 dynamic quantization fails the accuracy gate by a wide margin: 74% argmax agreement with fp32, max abs dp 0.92 (see [Quantization](#int8-dynamic-quantization)). |
| `LAYA_MCP_SYNC_ROW_BUDGET` | **20** | 20 (was 40) | 40 rows of long states is ~65 s of inference, too slow for an interactive tool call. 20 rows is ~9 s for short states, ~18 s for medium and ~32 s for long. |
| `LAYA_MCP_MAX_WAITING_REQUESTS` | **4** | 16 | With 16 queued calls of ~20 s each, the ones at the back time out after 120 s having done nothing. With 4, the worst-case wait (~4 x 20-30 s) stays near `request_timeout_s`, and later callers get a fast, retryable `EngineBusy` with an ETA. |
| `LAYA_MCP_REQUEST_TIMEOUT_S` | 120 | 120 | Keep. |
| `LAYA_MCP_MAX_LOADED` | 2 | 2 | english fp32 is ~2.0 GB RSS. english + multilingual should come to ~3.5 GB. |
| `LAYA_MCP_WARMUP` | true | true | The first pass after a load is slower; the warm-up absorbs it before traffic arrives. |

Config defaults are unchanged. These are recommendations for the deployment env file.

## Expected cost

Each question on each state is one **row**. All rows of a state go through one forward pass, and
each row is `[instructions + options]` (~40-120 tokens) plus the state, capped at 512 tokens for
the english checkpoint. Rule of thumb at 6 threads, from these measurements:

```
seconds per state ~= 0.25 + 0.003 x questions x tokens_per_row
```

| State (tokens) | tokens/row | ms per row (6 threads, 5-10 questions) | 5 questions | 10 questions |
|---|---|---|---|---|
| short (~40) | ~125 | ~460-530 | ~2.3 s | ~5 s |
| medium (~200) | ~285 | ~880 | ~4.4 s | ~9 s |
| long (~480, capped) | ~490-512 | ~1,600 | ~8 s | ~16 s |

With a single question the fixed overhead dominates: ~0.6 s short, ~1.3-1.7 s medium, ~2 s long.
One English token is ~4 characters, so a 1,000-character state is ~250 tokens. Anything past
~400-470 state tokens is cut off, depending on how long the question head is.

Batch throughput (one worker, 6 threads, 5 questions per item): ~1,500 items/hour for short states,
~800/hour for medium, ~450/hour for long.

The engine's ETA (`estimate_seconds`) uses a rolling average of ms per row (EMA, alpha 0.2,
700 ms fallback before the first measurement). Mixed state lengths make it accurate only to about
2x; a token-aware estimate is listed under follow-ups.

## Threads

The full sweep (below) ran each thread count in sequence, so load drift from other services
skewed it. The deciding measurement ran 6 and 8 threads **interleaved**, 5 runs each, 5 questions:

| | 6 threads | 8 threads | 8 vs 6 |
|---|---|---|---|
| medium (~200 tokens) | 4,412 ms (4,218-4,456) | 4,111 ms (3,917-4,153) | 7% faster |
| long (~480 tokens) | 8,053 ms (7,731-8,215) | 7,389 ms (7,241-7,557) | 8% faster |

The host CPU was ~80% busy during this check, this process included. In the sequential sweep,
4 and 6 threads came out the same (63.4 s vs 63.9 s over all configs) and 8 threads looked 26%
faster (47.0 s). The interleaved check shows most of that gap was drift.

## fp32 sweep

Median of 3 runs per config after one warm-up per thread count, in ms per call, with ms per row
in parentheses. The "long" state here was 406 tokens (the filler text was too short; since fixed
in the script). Rows still reached 490 tokens, close to the 512 cap.

| state | tokens/row | questions | 4 threads | 6 threads | 8 threads |
|---|---|---|---|---|---|
| short (~40 tok) | 124 | 1 | 616 (616) | 620 (620) | 1538 (1538) |
| short (~40 tok) | 124 | 5 | 2810 (562) | 2321 (464) | 3607 (721) |
| short (~40 tok) | 124 | 10 | 5292 (529) | 5351 (535) | 3725 (372) |
| medium (~200 tok) | 284 | 1 | 1326 (1326) | 1723 (1723) | 1488 (1488) |
| medium (~200 tok) | 284 | 5 | 6466 (1293) | 6803 (1361) | 5302 (1060) |
| medium (~200 tok) | 284 | 10 | 12083 (1208) | 12209 (1221) | 8612 (861) |
| long (~480 tok) | 490 | 1 | 2270 (2270) | 4016 (4016) | 1752 (1752) |
| long (~480 tok) | 490 | 5 | 11999 (2400) | 11032 (2206) | 7287 (1457) |
| long (~480 tok) | 490 | 10 | 20566 (2057) | 19835 (1983) | 13704 (1370) |
| **sum of all configs** | | | **63.4 s** | **63.9 s** | **47.0 s** |

Cold load of the english checkpoint: 5.4 s with the OS file cache warm (~10 s from a cold disk,
earlier measurement). RSS after load is 2.04 GB, above the ~1.7 GB in the build plan.

## int8 dynamic quantization

Only `torch.ao.quantization.quantize_dynamic` on the **encoder's** `nn.Linear` layers works (the
same function the server uses, `laya_mcp.engine.runtime.quantize_dynamic_int8`). Quantizing the
whole model fails at inference: the decision head's `nn.TransformerEncoderLayer` fast path reads
`linear1.weight` directly (`AttributeError: 'function' object has no attribute 'device'`). The
head is a small share of the compute, so it stays fp32.

Speed at 8 threads, the fastest fp32 setting in the sweep (median ms per call):

| state | questions | fp32 | int8 per-tensor | speedup | int8 per-channel | speedup |
|---|---|---|---|---|---|---|
| short | 1 | 1538 | 405 | 3.80x | 360 | 4.27x |
| short | 5 | 3607 | 2454 | 1.47x | 1016 | 3.55x |
| short | 10 | 3725 | 4085 | 0.91x | 2038 | 1.83x |
| medium | 1 | 1488 | 1499 | 0.99x | 667 | 2.23x |
| medium | 5 | 5302 | 4657 | 1.14x | 2610 | 2.03x |
| medium | 10 | 8612 | 8735 | 0.99x | 5126 | 1.68x |
| long | 1 | 1752 | 958 | 1.83x | 980 | 1.79x |
| long | 5 | 7287 | 4567 | 1.60x | 4599 | 1.58x |
| long | 10 | 13704 | 9366 | 1.46x | 9247 | 1.48x |

The short-state fp32 cells at 8 threads are inflated by contention (compare the 4/6-thread
columns), which exaggerates those speedups. The long-state rows, ~1.5x, are the most trustworthy.

Accuracy against fp32 on a fixed probe set of 32 (state, questions) pairs, 8 each from the
`triage`, `email`, `guard` and `router` presets, 152 answers in total:

| variant | argmax agreement | max abs dp | mean abs dp | p95 abs dp | speedup (geomean of sweep) | probe-set time | RSS | verdict |
|---|---|---|---|---|---|---|---|---|
| fp32 | 100% | 0 | 0 | 0 | 1.00x | 70.3 s | 2.04 GB | baseline |
| int8 per-tensor | 73.0% | 0.943 | 0.245 | 0.650 | 1.42x | 39.7 s (1.77x) | 2.72 GB | **rejected** |
| int8 per-channel | 74.3% | 0.921 | 0.245 | 0.637 | 2.13x | 28.2 s (2.49x) | 2.59 GB | **rejected** |

By answer type: per-channel keeps 24/32 choice, 70/88 noul and 19/32 score answers; per-tensor
keeps 18/32, 76/88 and 17/32. The flips are not borderline cases. Examples: triage `intent`
technical_help -> other, `refund_requested` false -> true, `frustration` levels moving by one.
Quantization does not reduce RSS either, because the fp32 checkpoint is loaded first and the
allocator keeps the freed pages.

**Decision rule:** recommend quantization only if argmax agreement is 100%, max |dp| < 0.02 and
speedup >= 1.3x. Both variants meet the speed bar and fail the accuracy bar by ~45x, so
`LAYA_MCP_QUANTIZE` stays **off**. If someone enables it anyway, the engine uses per-channel
weights (`QUANTIZE_PER_CHANNEL = True`). Any schema run that way must be re-evaluated and
recalibrated with `laya_evaluate` / `laya_calibrate`.

## Variance

Other services kept 23-39% of the CPU busy at the start and end of the run. Examples of the noise:

- 8 threads, short state, 1 question: 1,538 ms, against 616-620 ms at 4 and 6 threads. That is
  contention, not a property of 8 threads.
- Medium state, 5 questions, 6 threads: 6,803 ms in the sequential sweep but 4,412 ms in the
  interleaved check ~20 minutes later.

Treat single cells as +-30-50%. Use interleaved comparisons for decisions and the rule of thumb
above for budgets. Re-run on a quiet host (e.g. at night) for clean absolute numbers.

## How the engine uses this

- One forward pass at a time. Interactive tool calls take priority over batch-job states, so an
  interactive call waits for at most the one batch state in flight (~2-16 s by the table above).
  A steady stream of interactive calls starves batch jobs; that is by design.
- `request_timeout_s` covers queueing plus inference. A pass that has started cannot be
  interrupted: it finishes in its thread and its result is discarded.
- A prediction that needs a checkpoint that isn't loaded pays the load (~5-10 s) inside its slot.
  Preload whatever the traffic needs.

## Reproduce

```
set HF_HUB_OFFLINE=1
venv\Scripts\python.exe scripts\bench.py --out data\bench-results.md --json data\bench.json
venv\Scripts\python.exe scripts\bench.py --threads 6 --no-quant --repeats 5      # fp32 only
```

The default run takes ~15 minutes. `--quant-threads`, `--quant-variants`, `--questions`, `--lengths`
and `--probe-limit` shrink it.

## Follow-ups (not done in this spike)

- **ONNX Runtime + DirectML on the RX 6600M**: time-boxed in the plan, not attempted. It is the
  only route to GPU inference on this AMD host and is worth a spike before buying hardware.
- **ONNX Runtime CPU**: graph-optimized fp32, and static int8 with calibration, which usually holds
  accuracy much better than dynamic quantization. Must pass the same probe-set gate.
- **Selective quantization**: MLP layers only, or SmoothQuant-style activation scaling. The large
  flips suggest activation outliers in ModernBERT. bf16 is not an option: Zen 3 has no native
  bf16 matmul.
- **Token-aware budget and ETA**: cost scales with tokens per row, not rows, so a 20-row call ranges
  from 9 s to 32 s. See the contract change proposed in the engine report
  (`estimate_seconds(rows, tokens_per_row=None)` and a seconds-based sync budget).
