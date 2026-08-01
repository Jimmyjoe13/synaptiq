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
| `--arm qem\|vector\|mem0\|both\|all` | Q-EM engine, vector top-k baseline, mem0 OSS SDK; `both` = qem+vector (historical default), `all` = the three |
| `--mem0-collection` | pgvector collection for the mem0 arm. **Must start with `mem0`** — reset drops every table carrying that prefix |
| `--top-k` | Candidate pool size for the vector **and mem0** arms before token budget truncation |
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

## 1bis. The `mem0` arm — comparing against the open-source reference

`--arm mem0` (or `all`) runs the [mem0](https://github.com/mem0ai/mem0) OSS SDK as a third
arm, inside this harness. It exists because the published numbers on either side are **not
comparable**: different judge, different answering model, different question subset. The
only defensible figure is both engines measured under one protocol.

```bash
pip install 'mem0ai[nlp]'
python -m spacy download en_core_web_sm     # see the trap below — verify it landed

python benchmarks/locomo_runner.py benchmarks/locomo10.json \
    --conv 0 --arm all --top-k 50 --qa-workers 4 --out results.json
```

### What is held equal

Embedding model, extraction LLM, answering model, judge, PostgreSQL server, HNSW indexing,
ingested text (the exact same `[date] speaker: text` string, same order), and the token
budget — truncated by `benchmarks/budget.py`, the single implementation shared by the
vector and mem0 arms, built on the same `estimate_tokens` the Q-EM collapse uses.

### What stays asymmetric — quote it with any result

1. **Dating.** SynaptiQ also receives the session date as `created_at`, which feeds temporal
   decay. mem0 only gets it inline in the text and as metadata. Structural advantage to
   SynaptiQ on the `temporal` category.
2. **This measures the OSS SDK**, not mem0's managed platform — whose LoCoMo scores include,
   per their own note, "proprietary optimizations not available in the open-source SDK".
   The platform is not self-hostable, so it is out of SynaptiQ's product scope.
3. **Latency is not comparable.** mem0 v3 extracts inside `add()`; SynaptiQ consolidates
   asynchronously behind the outbox. LLM call counts are comparable, perceived latency is not.
4. **`en_core_web_sm` is an English model.** Correct for LOCOMO — and a reminder that this
   harness says nothing about mem0 on a French corpus, where SynaptiQ runs a multilingual
   embedder.

### Three traps, all found by actually running it

| Trap | Symptom | Guard |
|---|---|---|
| **`OPENROUTER_API_KEY` in the environment** | `mem0/llms/openai.py` checks that variable *before* reading the config: mem0's extraction silently goes to OpenRouter while SynaptiQ extracts locally. Here the local model name did not exist upstream, so it failed loudly — with a model name valid on both sides, the run would have completed and the report would have claimed "same extraction LLM" in good faith. | The arm pops the variable from the benchmark process and pins `OPENAI_BASE_URL`; `stats()["env_neutralized"]` records it |
| **`python -m spacy download en_core_web_sm` installing elsewhere** | Reports "Download and installation successful", yet `spacy.load` fails. mem0 then degrades to raw text: no BM25 lemmatization, no entity linking, **no error** — two of v3's three recall signals gone. | `stats()["nlp"]` reports the real state and the runner logs a loud warning. Fix: `pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"` |
| **mem0 v3 creates a second table** (`<collection>_entities`) | Resetting only the main table leaves the previous run's entities behind while memories start empty — an inconsistent state that raises nothing. | Reset sweeps by prefix, which is why the collection name must start with `mem0` |

`mem0ai[nlp]` only installs spaCy — BM25 is implemented inside mem0 itself.

### Report additions

`mem0` block (version, NLP capabilities, models, `llm_base_url` actually used, stored
memories, add/search failures) and a `comparisons` block holding every pairwise delta with
its confidence interval and significance verdict. `delta_qem_minus_vector` is kept for
backward compatibility.

> [!WARNING]
> `Difference` treats the arms as **independent** samples although they answer the same
> questions. That is conservative — a paired test yields a narrower interval — so it never
> favours an arm, but a delta declared non-significant here could be significant under the
> paired test that actually applies.

An ingestion failure ratio above `--max-degraded` aborts the run for the mem0 arm exactly as
degraded regex extractions do for SynaptiQ: a corpus with holes yields a low score that
measures the outage, not the engine.

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
