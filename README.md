---
title: TRAPLAB
emoji: 🧪
colorFrom: orange
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: space/app.py
pinned: false
license: mit
---

# TRAPLAB

**Can an adaptive trap designer find a model's statistical reasoning errors faster
than random search?** Beam proposes numerical configurations. Python alone writes
the question and computes its answer from the exact displayed information.

The first working release includes Simpson's paradox, collider selection and
regression to the mean; a strict validity gate; a resumable two-arm runner; raw
response logs; deterministic replay; and a read-only Gradio Failure Atlas.
The bundled dashboard uses **synthetic demo fixtures, not Beam results**.

## Quick start — no credential needed

```bash
cd /workspace/TRAPLAB
uv sync --frozen --extra space --extra publish
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/traplab smoke --demo --output artifacts/runs/smoke-demo
.venv/bin/traplab run --demo --config configs/demo.json --output artifacts/runs/demo
.venv/bin/python reproduce.py artifacts/runs/demo
.venv/bin/traplab extensions --demo --source artifacts/runs/demo \
  --output artifacts/runs/demo-extensions --bases 2 --self-limit 4 --flip-configs 1
.venv/bin/python space/app.py
```

Core generation, testing and replay also work without installation:
`PYTHONPATH=src python -m unittest discover -s tests -v`.
The app reads `demo-results/` and `demo-extensions/` by default; set
`TRAPLAB_RESULTS` and `TRAPLAB_EXTENSIONS` to matching main and extension runs
to inspect real results. It makes no inference calls and needs no key.
Verify the actual UI with `.venv/bin/python scripts/smoke_ui.py`.

## Live Beam smoke test — do this before any experiment

Use your provider's documented OpenAI-compatible API base URL and set
`BEAM_BASE_URL` and `BEAM_API_KEY` securely in environment settings or your shell.
The model defaults to `reflection/Beam-1.0-501B-A23B`; `BEAM_MODEL` can override it.
The client appends `/chat/completions` to the base URL. Do not put a key in a URL,
commit it, or paste it in chat. `.env.example` is documentation; `.env` files are
not automatically loaded. No specific provider endpoint has been verified yet.

```bash
.venv/bin/traplab smoke --output artifacts/runs/smoke-live
```

This sends one solver question, persists the response, and recomputes the grade.
A wrong response is a recorded reasoning error; it does not mean the transport
failed. Truncated and invalid JSON outputs are separate operational outcomes.
The smoke fixture uses a fixed local story; the experiment generates a shared
story pool with Beam:

```bash
.venv/bin/traplab stories --output artifacts/runs/story-pool
.venv/bin/traplab run --config configs/pilot.json \
  --stories artifacts/runs/story-pool/pool.json --output artifacts/runs/pilot
# Only after assessing the pilot, freezing the protocol, and checking cost:
.venv/bin/traplab run --config configs/main.json \
  --stories artifacts/runs/story-pool/pool.json --output artifacts/runs/main
```

Main includes 360 solver evaluations plus designer calls and possible length
retries. Generation is billable. No experiment runs automatically. Resume by
repeating exactly the same command; completed responses are reused. Source/config
changes require a new directory. Use a single process per run directory.
Keep `calls.jsonl` beside the story-pool file. The runner imports those shared
generation calls once into its ledger, so reported usage includes the pool cost.
If combining multiple reports sharing a pool, do not charge that shared cost twice.

Run the registered supplementary experiments only after the main run completes:

```bash
.venv/bin/traplab extensions --source artifacts/runs/main \
  --output artifacts/runs/extensions
.venv/bin/python reproduce.py artifacts/runs/extensions
```

Defaults: 20 bases × 5 stories × 3 repeats; up to 60 wrong + 60 correct
self-correction items; two extra calls on a seeded 10% sample; and 3 × 11 × 3
Simpson flip-grid calls, excluding invalid ambiguity points. No noisy binary
search is used. Fixed grids and unresolved/no-flip/unstable outcomes are explicit.
Self-correction is descriptive because it has no no-tool control.

## Audit and publish

```bash
.venv/bin/python reproduce.py artifacts/runs/main
.venv/bin/python scripts/publish_dataset.py artifacts/runs/main artifacts/dataset-bundle
```

Packaging is local by default. The bundle declares dataset configs `items`,
`traces`, `rejected`, `stories` and retains the call ledger and manifest. Inspect
traces and credentials hygiene before public upload. An explicitly requested
upload uses `--upload YOUR-HF-NAME/traplab` and an authenticated Hugging Face
session. Nothing has been pushed or published by this initial implementation.
For a Space, upload this repository as a public Gradio Space; root metadata and
requirements configure `space/app.py`. No API key belongs in the read-only Space.

Read [the protocol](protocol/v1.md) for causal assumptions, ambiguity bands,
binning, rejection policy, inference settings and statistical limitations.
Self-correction, repeated-item noise, story sensitivity and flip-point experiments
are implemented and tested offline; none has run against Beam yet.

## Artifact contract

Each run contains `manifest.json`, `items.jsonl`, `traces.jsonl`, `calls.jsonl`,
`stories.jsonl`, and any proposal/rejection/event ledgers. `summary.json` is a
derived convenience file. The manifest records source hashes, model, seeds,
configuration, timestamp, story pool and Git commit where available. Raw files
are append-only. `reproduce.py` reconstructs questions and grades and rejects
mismatches before computing metrics. It requires no provider access.

Demo answers deliberately make mistakes to exercise the interface and pipeline.
They are not an assessment of Beam, nor evidence for the research hypothesis.
