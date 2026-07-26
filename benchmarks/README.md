# SynaptiQ Benchmarks

Two harnesses for two use cases: an **offline micro-benchmark** of Q-EM greedy collapse, and an **end-to-end benchmark** on LOCOMO measuring the full pipeline (ingestion → consolidation → recall → response generation) against a standard vector baseline.

---

## 1. LOCOMO — Conversational Long-Term Memory

`locomo_runner.py` replays multi-session dialogue across the **real pipeline** (including the extraction worker), answers dataset questions, and evaluates responses via an LLM judge. This follows the dominant evaluation protocol in the literature ("J-score").

### Dataset

The dataset is **not versioned in this repository** as it is third-party work distributed under its own license. Download it directly from the official repository:

```bash
curl -L -o benchmarks/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

10 conversations, ~5,900 turns, ~1,990 questions across 5 categories (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial — **excluded** by convention, matching standard research methodology).

### Execution

```bash
python benchmarks/locomo_runner.py benchmarks/locomo10.json \
    --conv 0 --arm both --top-k 50 --qa-workers 4 --out results.json
```

| Option | Role |
|---|---|
| `--arm qem\|vector\|both` | Q-EM engine, vector top-k baseline, or both |
| `--top-k` | Candidate pool size for vector baseline before token budget truncation |
| `--qa-workers` | Parallel QA evaluation workers (ingestion remains sequential to build graph edges) |
| `--resume` | Resumes interrupted ingestion instead of starting over from scratch |
| `--max-degraded` | Maximum allowed ratio of regex fallback extractions before aborting (default 5%) |
| `--limit-turns`, `--limit-qa` | Limits for quick smoke testing |

### Harness Guarantees

- **Equal Token Budget Comparison.** Both arms are truncated under the exact same token budget using the **same estimator** (`estimate_tokens`). Without this, accuracy deltas would reflect context length rather than retrieval quality.
- **Independent LLM Judge.** `LOCOMO_MODEL_QA` and `LOCOMO_MODEL_JUDGE` are separated to eliminate self-preference bias. The QA model is strictly identical for both arms.
- **Degraded Extraction Safeguard.** When an LLM fails (rate-limit, timeout), the worker falls back to regex heuristics, producing `episodic` memories excluded from graph entanglement. The harness counts these fallbacks (`degraded_ratio`) and **aborts the run** if the threshold is exceeded.
- **Reproducibility.** Generated JSON reports include all active LLM models, embedding models, and Q-EM thresholds.

---

## 2. Antigravity Shim — `agy_openai_shim.py`

Exposes a local **OpenAI-compatible endpoint** relaying requests to Antigravity CLI (`agy`), authenticating via machine credentials without per-minute API quotas.

```bash
python benchmarks/agy_openai_shim.py --port 8899 --model gpt-oss-120b-medium
# then in .env: LLM_BASE_URL=http://127.0.0.1:8899/v1
```

SynaptiQ communicates strictly via standard OpenAI-compatible HTTP endpoints: **no production code is vendor-locked**.

---

## 3. Micro-benchmark — `qem_vs_vector.py`

Compares offline Q-EM collapse against a top-k baseline of equal token size on pre-scored candidate pools. No database infrastructure required.

```bash
PYTHONPATH=packages/core python benchmarks/qem_vs_vector.py dataset.jsonl
```

---

## Publishing Benchmark Results

A score report must always include the **dataset, embedding model, LLM models, and active Q-EM thresholds** — all included automatically in the generated JSON report output.
