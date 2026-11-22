<div align="center">

<p align="center">
  <img src="img/logo-synaptiq3.png" alt="SynaptiQ" width="" />
</p>

<h3><em>Long-term memory infrastructure for AI agents — semantic, temporal, self-structuring.</em></h3>

<p>
  <strong>Q-EM</strong> · <em>Quantum-like Entanglement Memory</em> — a memory engine that links
  memories by <strong>meaning</strong>, not by hand-written wikilinks.
</p>

<p>
  <a href="https://github.com/Jimmyjoe13/synaptiq/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Jimmyjoe13/synaptiq/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white">
  <img alt="Redis" src="https://img.shields.io/badge/Redis-Streams-DC382D?logo=redis&logoColor=white">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-ready-6E56CF">
  <img alt="Coverage" src="https://img.shields.io/badge/core%20coverage-96%25-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p>
  <a href="#what-it-is">What it is</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#core-concepts">Concepts</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#sdks">SDKs</a> ·
  <a href="#mcp-server">MCP</a> ·
  <a href="docs/agent-integration.md">Agent integration</a> ·
  <a href="#security">Security</a> ·
  <a href="#benchmarks">Benchmarks</a>
</p>

</div>

---

## What it is

**SynaptiQ is long-term memory for autonomous agents.** Where a standard RAG stack retrieves
the *k* nearest raw chunks, SynaptiQ **consolidates** an agent's stream of experience into
discrete memories, **links** them semantically, **prunes** contradictions and redundancies,
and **assembles** a compact context packet that fits a token budget.

It runs on your own hardware: PostgreSQL + pgvector for storage, Redis Streams for async
ingestion, FastAPI for the surface, and a native MCP server for agent clients.

> [!NOTE]
> **Self-hosted: one deployment = one security perimeter.** The tenant boundary is decided
> server-side and never trusted from a request payload. Agents inside a deployment are
> isolated by `agent_id`.

**Project status.** Actively developed, running in production on a single-instance
deployment. The API is versioned under `/v1`; the schema is owned by Alembic migrations.
Breaking changes are documented in [`CHANGELOG.md`](CHANGELOG.md). The comparative benchmark
is honest about not yet being statistically significant — see [Benchmarks](#benchmarks).

### When SynaptiQ fits — and when it doesn't

| Good fit | Poor fit |
|---|---|
| An agent that must remember across sessions and **stop repeating its mistakes** | One-shot document Q&A over a static corpus — use a plain vector store |
| Knowledge that **evolves and contradicts itself** (preferences, rules, decisions) | Immutable reference documents |
| A **token-budgeted** context assembled per task | Feeding an entire corpus into a long context window |
| Self-hosting, data staying on-premise | A fully managed SaaS you don't want to operate |

<br>

## Quickstart

**Prerequisites** — Docker & Docker Compose · an embedding provider ([LM Studio](https://lmstudio.ai)
locally, or an OpenRouter/OpenAI key) · Python 3.11+ for non-container development.

```bash
git clone https://github.com/Jimmyjoe13/synaptiq.git
cd synaptiq
cp .env.example .env       # review EMBEDDING_PROVIDER and the API keys
docker compose up -d       # Postgres + Redis + migrate + API + worker + relay
```

```bash
curl http://127.0.0.1:8000/v1/health
# {"status":"ok","services":{"postgres":"healthy","redis":"healthy","ingestion":"healthy"}}
```

| Service | Endpoint | Started by default |
|---|---|:---:|
| API (`/v1`) | `http://127.0.0.1:8000` | ✅ |
| PostgreSQL | `127.0.0.1:5435` | ✅ |
| Redis | `127.0.0.1:6399` | ✅ |
| Worker + relay | *(no port)* | ✅ |
| MCP server | `http://127.0.0.1:8765` | ⛔ `--profile mcp-http` |

The MCP service sits behind a Compose profile: `docker compose --profile mcp-http up -d`.
See the [MCP guide](docs/mcp-server.md).

<details>
<summary><strong>Running the Python services on the host instead</strong></summary>

<br>

Needed when your embedding server only listens on loopback — a container cannot reach it.

```bash
docker compose up -d postgres redis migrate    # storage + schema only

pip install -r requirements-dev.txt
pip install -e packages/core -e packages/sdk-python

python -m uvicorn apps.api.main:app --reload --port 8000
python apps/worker/worker.py        # separate terminal
python apps/relay/relay.py          # separate terminal — publishes the outbox
```

Skipping the **relay** is the classic mistake: `/events` still answers `201`, the event lands
in the transactional outbox, and nothing ever consolidates it. `/v1/health` reports
`"ingestion":"stalled"` when that happens.

</details>

### First memory, end to end

```bash
# Write a consolidated memory directly
curl -X POST http://127.0.0.1:8000/v1/memories -H 'Content-Type: application/json' -d '{
  "agent_id": "george", "compartment": "reflexes",
  "content": "Jimmy prefers short progress reports, in French." }'

# Ask for a context packet before calling your LLM
curl -X POST http://127.0.0.1:8000/v1/context/build -H 'Content-Type: application/json' -d '{
  "agent_id": "george", "session_id": "s1",
  "task": "Draft the weekly update", "query": "writing style preferences" }'
```

<br>

## Architecture

The calling agent never waits for heavy processing: extraction, embedding and graph building
are offloaded to a background worker.

```mermaid
graph LR
    A[🤖 Agent] -->|1. POST /events| API[⚙️ FastAPI]
    API -->|2. same transaction| OB[(📥 event_outbox)]
    OB -->|3. relay publishes| R[(📨 Redis Streams)]
    R -->|4. consumer group| W[🧠 Worker]
    W -->|5. extract · embed · entangle| DB[(🗄️ PostgreSQL + pgvector)]
    A -->|6. POST /context/build| API
    API -->|7. Q-EM recall| DB
    DB -->|8. context packet| API
    API -->|9. rehydrated prompt| A
```

| Component | Path | Role |
|---|---|---|
| **API** | `apps/api` | Ingestion, retrieval, context assembly, compartment routing. |
| **Relay** | `apps/relay` | Publishes the transactional outbox to Redis. Without it, `/events` is a silent black hole. |
| **Worker** | `apps/worker` | Consumes the stream: extract → embed → entangle. Consumer group, ACK, bounded retries, DLQ. |
| **Core** | `packages/core` | Pure Q-EM engine, pluggable `Embedder`, governance, the 9 compartments. Zero I/O, 96% covered. |
| **MCP** | `apps/mcp` | Seven tools over MCP (`stdio` / `http`). |
| **SDKs** | `packages/sdk-python`, `packages/sdk-typescript` | Native clients. |

**Write reliability.** `/events` writes the event *and* its outbox row in a single
transaction — never straight to Redis. The relay publishes, the worker deduplicates on
`memories.source_event_id`, and `idempotency_key` makes a repeated submission a no-op. The
guarantee is at-least-once delivery without duplicated memories.

<br>

## Core concepts

### The Q-EM engine, in four phases

A quantum-inspired metaphor over a strictly deterministic pipeline. Every
`POST /v1/context/build` runs:

```
  1. SUPERPOSITION    Hybrid search: pgvector similarity ∪ full-text, fused by RRF
        │             initial score = relevance × recency_factor
        ▼
  2. ENTANGLEMENT     Damped activation spreading along 'entangled_with' edges (multi-hop)
        │             → related memories surface even without matching the query text
        ▼
  3. INTERFERENCE     Destructive filtering:
        │               • contradictions → the superseded version is cancelled
        ▼               • redundancies (cosine > threshold) → one survivor
  4. MEASUREMENT      Greedy collapse by utility density per token, routed into sections
```

The algorithmic core lives in `packages/core/synaptiq_core/qem.py`: **pure functions, no I/O,
fully unit-tested.** Every threshold is an environment variable, read at call time, so an
ablation study needs no redeploy.

### Compartments: a memory organised like a brain

Since v0.4 a memory lives in one of **nine fixed compartments**, each modelled on a region of
the human brain and each carrying its own **behaviour**. The agent creates nothing: it only
judges where a memory belongs. (Until v0.3 agents created their own collections; a production
audit found 80 of them, a third holding a single memory, near-duplicates everywhere, and the
same collection spread across three families — hence three different behaviours.)

| Compartment | Brain region | What goes there | Behaviour |
|---|---|---|---|
| `reflexes` | Brain stem | Hard rules, firm user preferences, guardrails | **Injected into every context**, never forgotten, ≤ 20 % of the budget |
| `decisions` | Frontal lobe | Decisions and their rationale, goals, plans | Slow forgetting, entangled |
| `savoir_faire` | Cerebellum | Procedures, playbooks, coding conventions | Very slow forgetting, entangled |
| `alertes` | Amygdala | Resolved errors, incidents, traps not to repeat | Weighted ×1.15 at recall, slow forgetting |
| `environnement` | Parietal lobe | Map of the terrain: servers, access, paths, ports | Slow forgetting, entangled |
| `savoirs` | Temporal lobe | Durable facts, project state, definitions | Medium forgetting, entangled |
| `episodes` | Hippocampus | Session log, interactions, milestones | Fast forgetting, outside the graph |
| `observations` | Occipital lobe | Raw findings: measurements, audit results, logs | Fastest forgetting, outside the graph |
| `croyances` | Prefrontal cortex | The agent's hypotheses about the user and people | Never decays, rendered as *"(hypothesis, confidence x)"*, ≤ 15 % of the budget, written only through `note_belief` |

```bash
curl -X POST http://127.0.0.1:8000/v1/memories -H 'Content-Type: application/json' -d '{
  "agent_id": "george", "compartment": "alertes",
  "content": "pip resolved starlette too far; pin fastmcp in the same install command." }'
```

**Filing never leads nowhere.** `compartment` is optional: when it is missing or unknown, the
engine classifies the memory by similarity with each compartment's prototype (its description
and examples, embedded with the instance's own model) — and the response always says where the
memory went and who decided (`compartment_origin`: `agent`, `alias`, `moteur`, `existant`).
An old v0.3 collection name (`preference`, `code_error_resolution`…) is converted through a
fixed alias table, so existing clients keep working during the transition.

> [!IMPORTANT]
> **`context_packet` has a fixed shape**: nine sections, always present, in the order above,
> even when empty. The internal family (`memories.type`) is now *derived* from the compartment
> and no longer accepted from callers. `GET /v1/compartments` lists the compartments with the
> agent's memory count in each.

Migrating an existing instance: the `20261122_compartments` migration converts every memory
with a frozen copy of the alias table (keeping the old name in `legacy_subtype`), then
`scripts/reclassify_compartments.py` proposes a similarity-based compartment for the memories
the table could only file by family — as a CSV to review before `--apply`.

### Contradictions

Similarity alone never archives anything. An earlier implementation treated "close" as
"contradictory" and silently destroyed compatible preferences — *"short emails"* and *"emails
in French"* sit around 0.85 cosine. Today the cosine is only a **pre-filter** that bounds how
many candidates reach a pluggable judge; archiving requires an **explicit verdict**, and the
judge is fail-closed (no judge configured → nothing is archived). A `supersedes_by` edge
records what replaced what, so an archive is never indistinguishable from a disappearance.

<br>

## API reference

Base path `/v1`. Unversioned aliases are kept for backward compatibility.

| Method | Endpoint | Role |
|:---:|---|---|
| `GET` | `/v1/health` | Postgres, Redis and **ingestion** status. `degraded` if any is unhealthy. |
| `POST` | `/v1/events` | Async ingestion, transactional outbox, idempotent via `idempotency_key`. |
| `POST` | `/v1/memories` | Direct write of a consolidated memory. **Idempotent on content** — a retry returns the same `memory_id` with `status: "duplicate"`. Returns the resolved `compartment` and who chose it. |
| `POST` | `/v1/retrieve` | Hybrid search (vector + full-text, RRF), filterable by compartment. |
| `POST` | `/v1/context/build` | Q-EM context packet under a token budget. `explain=true` adds a `retrieval_trace`. |
| `GET` | `/v1/compartments` | The nine compartments, their behaviour and the agent's memory count in each. |
| `GET` | `/v1/beliefs` · `POST` `/v1/beliefs/{id}/contest` | What the agent thinks of the user — readable and contestable. |
| `DELETE` | `/v1/memories` | GDPR purge — `admin` scope **and** `?confirm=<tenant_id>`. Optional `?agent_id=`. |
| `GET` | `/metrics` | Prometheus exposition. |

Interactive schema at `http://127.0.0.1:8000/docs`.

<br>

## SDKs

### Python — `synaptiq-sdk`

```python
from synaptiq_sdk import SynaptiqClient

client = SynaptiqClient("http://127.0.0.1:8000", api_key="your_api_key")

# Capture a raw interaction — consolidated asynchronously by the worker
client.capture(agent_id="george", session_id="s1",
               content="User prefers concise progress reports in English.")

# File a memory in a compartment (omit it and the engine classifies)
client.store_memory(agent_id="george", compartment="savoir_faire",
                    content="Run the full test suite before any deployment.")

# Rehydrate a compact context packet before calling the LLM
ctx = client.build_context(agent_id="george", session_id="s1",
                           task="Draft the weekly update",
                           query="format and language preferences")

for section, items in ctx["context_packet"].items():   # the nine compartments, in order
    for item in items:
        print(f"[{section}] {item}")
print(ctx["token_estimate"])
```

### TypeScript — `@synaptiq/sdk`

```typescript
import { SynaptiqClient } from "@synaptiq/sdk";

const client = new SynaptiqClient({
  baseUrl: "http://127.0.0.1:8000",
  apiKey: "your_api_key",
});

await client.capture("george", "s1", "User prefers concise progress reports in English.");

const ctx = await client.buildContext("george", "s1",
  "Draft the weekly update", "format and language preferences",
  { maxTokens: 1200 });

for (const [section, items] of Object.entries(ctx.context_packet)) {
  for (const item of items) console.log(`[${section}] ${item}`);
}
```

<br>

## MCP server

Seven tools, exposed to Claude Desktop, Cursor, Codex CLI, antigravity CLI, or any MCP client:

| Tool | Purpose |
|---|---|
| `store_memory` | Write a memory into a compartment — and report where it landed and who decided. |
| `list_compartments` | The nine compartments, their behaviour and volumes. |
| `recall_memories` | Hybrid search, filterable by compartment. |
| `build_context` | Assemble a Q-EM packet under a token budget, reflexes always included. |
| `note_belief` · `list_beliefs` · `contest_belief` | The agent's hypotheses about the user, under guardrails. |

```json
{ "mcpServers": { "synaptiq": { "serverUrl": "http://127.0.0.1:8765/mcp/" } } }
```

No tool takes `agent_id`. Identity comes from `SYNAPTIQ_AGENT_ID` in the server environment,
so no prompt can make the model act as another agent.

📖 **[Full MCP guide](docs/mcp-server.md)** — transports (`stdio` vs `http` and their measured
trade-off), Docker and host installs, client configuration, verification, troubleshooting.

📖 **[Agent integration guide](docs/agent-integration.md)** — the other half of the job: how to
make an agent actually *use* the memory well. Filing in the right compartment, writing memories that
survive being recalled alone, `build_context` vs. raw search — and the four asymmetries that
degrade recall in silence, starting with the one that matters most: **a direct write to
`/v1/memories` builds no `entangled_with` edges**, so an agent that only calls `store_memory`
never builds a graph and loses the multi-hop phase entirely.

<br>

## Configuration

All settings are environment variables in the repository-root `.env`
([`.env.example`](.env.example) is the annotated template).

```env
EMBEDDING_PROVIDER=lmstudio                 # lmstudio | openrouter | openai | mock
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf
EMBEDDING_DIM=384                           # must match the VECTOR(n) column

SYNAPTIQ_TENANT=default
SYNAPTIQ_AUTH_REQUIRED=true
```

📖 **[Full configuration reference](docs/configuration.md)** — storage, embeddings, LLM
extraction, retrieval, Q-EM thresholds, compartments, ingestion, security, observability.

> [!CAUTION]
> **Never change `EMBEDDING_MODEL` on a populated instance without re-embedding.** Two
> *different* models of the *same* dimension raise no error at all — the stored vectors just
> stop being comparable and recall degrades in silence. The worker refuses to start on such a
> mismatch (`EMBEDDING_COHERENCE_CHECK`).

<br>

## Security

Tenant scoping is server-side (`SYNAPTIQ_TENANT`) and **never** read from a client payload.
Agents within a tenant are isolated by `agent_id` — including their taxonomy.

```bash
python scripts/create_api_key.py --name "agent-prod"                       # read + write
python scripts/create_api_key.py --name "dashboard" --scopes read          # read-only
python scripts/create_api_key.py --name "agent-A" --agents agentA          # one agent only
python scripts/create_api_key.py --name "ops" --scopes read write admin    # may purge
```

Keys are stored as SHA-256 hashes and shown once. Send them as `Authorization: Bearer <key>`.

- **`agent_id` is enforced, not advisory.** A key created with `--agents agentA` gets `403`
  when acting as `agentB`. Without `--agents`, it reaches every agent of its tenant.
- **The GDPR purge needs `admin` *and* `?confirm=<tenant_id>`**, and writes an `audit_log` row
  in the same transaction. No purge without a trace, no trace without a purge.
- **`SYNAPTIQ_AUTH_REQUIRED=false` does not open the purge.** It is a convenience for a
  trusted localhost instance; `admin` still requires a key. A convenience flag must never
  unlock an irreversible endpoint.
- **`SYNAPTIQ_AGENT_ID` has no default.** It once had one, and an instance whose memories were
  written under another identity answered *"no memory found"* — with no error. For a memory
  engine, that symptom is indistinguishable from an empty store.

Vulnerability reports: [`SECURITY.md`](SECURITY.md).

<br>

## Observability

- **`/metrics`** (Prometheus): event counters, context-build latency histogram, auth cache
  hit ratio, degraded extractions — plus three gauges that predict incidents rather than
  merely count them: `synaptiq_outbox_pending`, `synaptiq_outbox_oldest_age_seconds`,
  `synaptiq_dlq_depth`.
- **`synaptiq_graph_edges_per_memory{agent_id}`** — entanglement density, per agent. An agent
  sitting at `0` with hundreds of memories has no graph at all, so the multi-hop phase of Q-EM
  runs empty and recall degrades **in silence**. That state went unnoticed for weeks on a real
  instance for want of this one number. Fix it with
  [`scripts/rebuild_entanglement.py`](scripts/rebuild_entanglement.py).
- **`synaptiq_memory_writes_total{outcome}`** — `created` vs. `duplicate` on direct writes.
  Without it, idempotency is unverifiable from the outside: a no-op and a creation look alike
  to the caller.
- **`/v1/health`** reports `ingestion` alongside Postgres and Redis. A dead relay makes
  `/events` a silent black hole — accepted, persisted, never consolidated — so the check lives
  where people actually look.
- **Structured logs** (`LOG_FORMAT=json`) with a `trace_id` propagated through a contextvar
  across the API and the core, and returned to the client. Every error path carries
  `exc_info`.

<br>

## Benchmarks

### LOCOMO — Q-EM vs. a vector baseline

One conversation: 419 dialogue turns, 152 evaluated questions, fixed 1500-token budget,
identical answering model on both arms.

| Category | Vector baseline (top-k) | SynaptiQ (Q-EM) | Difference |
|:---|:---:|:---:|:---:|
| **Overall accuracy** | 48.03 % | 51.32 % | +3.29 pts · 95% CI **[−7.9, +14.5]** |
| Multi-hop (graph) | 18.75 % | 25.00 % | +6.25 pts |
| Temporal reasoning | 78.38 % | 83.78 % | +5.40 pts |
| Single-hop | 47.14 % | 52.86 % | +5.72 pts |

> [!WARNING]
> **Read this before quoting those numbers. None of these differences is statistically
> significant.** On 152 questions, the 95% confidence interval on the overall gain spans
> −7.9 to +14.5 points — it contains zero, so this run **cannot** establish that Q-EM beats
> the baseline. Per-category intervals are wider still (multi-hop rests on 16 questions).
>
> We publish it anyway, with its uncertainty, because that is what the measurement says. The
> direction is consistent with what the algorithm does — the largest gains land on multi-hop
> and temporal questions, exactly what the entanglement graph and `occurred_at` are for — but
> *consistent* is not *demonstrated*.
>
> A ±2 point margin needs ~2,400 questions, i.e. the full 10-conversation LOCOMO set. That run
> is the next milestone; until it lands, treat this table as a smoke test, not as proof.
>
> Intervals are Wilson intervals computed by `synaptiq_core.stats` and emitted by the harness
> itself, so a result can no longer be published without its uncertainty.

Run details: 1,420 entanglement edges built by the worker, **0.0 % degraded extractions**
(every memory came from structured LLM extraction, none from the regex fallback — a run with
degraded extractions measures a handicapped Q-EM). Fixed seed, dedicated tenant. Reproduce
with `make bench` (`.\scripts\dev.ps1 bench` on Windows).

### Knowledge evolution vs. static Markdown search

Two scenarios where stored knowledge is later contradicted:

| Scenario | Obsidian MCP (Markdown) | SynaptiQ (Q-EM) |
|:---|:---:|:---:|
| DB migration contradiction | ❌ keeps both MySQL and Postgres | ✅ supersedes MySQL (`supersedes_by`) |
| Style rule update | ❌ returns contradictory advice | ✅ filters the obsolete rule out |

Two scenarios is a demonstration of the mechanism, not a measurement — the sample is far too
small to generalise.

<br>

## Development

```bash
# Unit tests — no infrastructure required
pytest tests/unit

# Full suite — needs Postgres + Redis
docker compose up -d postgres redis migrate
EMBEDDING_PROVIDER=mock \
DATABASE_URL=postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db \
REDIS_URL=redis://127.0.0.1:6399/0 \
pytest tests

# Lint, types, coverage
ruff check apps packages scripts tests benchmarks migrations examples
mypy
pytest tests/unit --cov=synaptiq_core --cov-fail-under=90
```

`make <target>` or `.\scripts\dev.ps1 <target>` (same names): `lint`, `types`, `test`,
`coverage`, `bench`, `bench-explain`.

**377 tests** (307 unit, 70 integration), core coverage **96 %**, mypy clean, ruff with
`E,W,F,I,B,UP,S,C4,RUF`. Anything outside `tests/unit/` is auto-marked `integration` and
requires Postgres + Redis.

The schema is owned **solely** by Alembic (`migrations/`). `infra/postgres/init.sql` only
creates the pgvector extension — never add DDL there.

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

<br>

## Roadmap

- Full 10-conversation LOCOMO run (~1,990 questions) to settle the significance question.
- Compose profiles `minimal` / `local-ai` (Ollama).
- Typed SDK errors and pagination; OpenTelemetry traces.
- Retention and export policies; LangGraph example.

<br>

---

<div align="center">

**MIT License** — see [`LICENSE`](LICENSE).

<sub>Built for agents that cannot afford to forget. 🧠</sub>

</div>
