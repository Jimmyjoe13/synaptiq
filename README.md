<!-- ╔══════════════════════════════════════════════════════════════╗ -->
<!-- ║                          SYNAPTIQ                              ║ -->
<!-- ╚══════════════════════════════════════════════════════════════╝ -->

<div align="center">

<p align="center">
  <img src="img/logo-synaptiq3.png" alt="SynaptiQ Logo" width="" />


<h3><em>The vector second brain for AI agents — long-term, semantic, and temporal memory.</em></h3>

<p>
  <strong>Q-EM</strong> · <em>Quantum-like Entanglement Memory</em> — a memory engine that connects memories by <strong>meaning</strong>, not manually written wikilinks.
</p>

<p>
  <a href="https://github.com/Jimmyjoe13/synaptiq/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Jimmyjoe13/synaptiq/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white">
  <img alt="Redis" src="https://img.shields.io/badge/Redis-Streams-DC382D?logo=redis&logoColor=white">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-ready-6E56CF">
  <img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p>
  <a href="#-why-q-em">Why Q-EM</a> ·
  <a href="#-key-features">Key Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-quickstart">Quickstart</a> ·
  <a href="#-the-q-em-engine-in-4-phases">Q-EM Engine</a> ·
  <a href="#-sdk-usage">SDKs</a> ·
  <a href="#-api-reference-v1">API</a> ·
  <a href="#-installing-the-mcp-server">MCP Server</a> ·
  <a href="#-security">Security</a>
</p>

</div>

---

**SynaptiQ** is a production-grade Long-Term Memory (LTM) infrastructure for autonomous AI agents. Unlike standard "flat top-k" RAG systems that simply retrieve the 5 nearest raw document chunks, SynaptiQ **consolidates** the agent's stream of experience, **entangles** memories semantically, **prunes** contradictions and redundancies, and **assembles** a compact context packet perfectly tailored to your token budget.

> [!NOTE]
> **Self-hosted: 1 deployment = 1 security perimeter.** The tenant boundary is enforced server-side and never trusted from request payloads. Keep your data on-premise while giving your agents a memory that learns from its mistakes.

<br>

## 💡 Why Q-EM

Standard RAG answers: *"Which chunks look similar to my query text?"*  
Q-EM answers: *"What exact context does the agent actually need right now to act without repeating itself?"*

| Feature / Dimension | 🗂️ Standard RAG | 🧠 SynaptiQ (Q-EM) |
|---|:---|:---|
| **Stored Unit** | Raw document chunks | Consolidated memories (facts, preferences, rules, errors…) |
| **Selection Algorithm** | Top-k cosine similarity | Superposition → **entanglement** → interference → measurement |
| **Related Memories** | Ignored if they don't match query text | **Retrieved via activation** along `entangled_with` graph edges |
| **Contradictions** | Returned side-by-side | **Adjudicated** — an obsolete memory is superseded only on an explicit contradiction verdict, never on similarity alone |
| **Redundancies** | Fill up context budget | **Pruned** (cosine > threshold → only one survives) |
| **Temporality** | Missing or naive | Configurable **temporal decay** (half-life, recency boost) |
| **Output** | Unstructured list of chunks | **Structured packet** under token budget (facts, preferences, rules, errors…) |

<br>

### 📊 LOCOMO Benchmark (Proof of Value vs Vector Baseline)

Run on **LOCOMO**, one conversation: 419 dialogue turns, 152 evaluated questions, fixed
context budget of 1500 tokens.

| Question Category | Vector Baseline (Top-k) | 🧠 SynaptiQ (Q-EM) | Difference |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | 48.03 % | 51.32 % | +3.29 pts · 95% CI [−7.9, +14.5] |
| **Multi-Hop** (graph entanglement) | 18.75 % | 25.00 % | +6.25 pts |
| **Temporal Reasoning** (timestamping) | 78.38 % | 83.78 % | +5.40 pts |
| **Single-Hop** | 47.14 % | 52.86 % | +5.72 pts |

> ### ⚠️ Read this before quoting the numbers above
>
> **On 152 questions, none of these differences is statistically significant.** The 95%
> confidence interval on the overall gain spans **−7.9 to +14.5 points** — it contains zero,
> so this run cannot establish that Q-EM beats the baseline. The per-category intervals are
> wider still (multi-hop rests on 16 questions).
>
> We publish it anyway, with its uncertainty, because that is what the measurement actually
> says. The direction is consistent with what the algorithm does — the largest gains land on
> multi-hop and temporal questions, which are exactly what the entanglement graph and
> `occurred_at` are for — but *consistent* is not *demonstrated*.
>
> Reaching a ±2 point margin needs **~2,400 questions**, i.e. the full 10-conversation LOCOMO
> set (~1,990 questions). That run is the next milestone; until it lands, treat this table as
> a smoke test, not as proof.
>
> Every interval here is computed by `synaptiq_core.stats` (Wilson intervals) and emitted by
> the harness itself, so a result can no longer be published without its uncertainty.

* 🔗 **1,420 entanglement relations** automatically generated by the background worker.
* 🛡️ **0.0 % degraded extractions** (every memory came from structured LLM extraction, none
  from the regex fallback — a run with degraded extractions measures a handicapped Q-EM).
* 🎲 Fixed seed, dedicated tenant, identical token budget and identical answering model on
  both arms. Reproduce with `make bench` (or `.\scripts\dev.ps1 bench` on Windows).

<br>

### ⚔️ SynaptiQ Q-EM vs Obsidian MCP (Static Markdown Search)

Comparative evaluation on knowledge evolution & contradiction resolution scenarios:

| Test / Scenario | 📝 Obsidian MCP (Markdown Notes) | 🧠 SynaptiQ (Q-EM) | Analysis & Victory |
|:---|:---:|:---:|:---|
| **DB Migration Contradiction** | ❌ Failed (0/1) | **✅ Passed (1/1)** 🥇 | Obsidian keeps MySQL & Postgres; SynaptiQ supersedes MySQL (`supersedes_by`). |
| **Style & Tone Rule Update** | ❌ Failed (0/1) | **✅ Passed (1/1)** 🥇 | Obsidian outputs contradictory advice; SynaptiQ filters out obsolete rules. |
| **Contradiction Resolution Rate** | **0.0 %** | **100.0 %** 🥇 | **Absolute supremacy of SynaptiQ on continuous learning.** |

<br>

## ⚡ Key Features

**🧠 Memory Engine**
- 🌌 **Q-EM Core** — semantic superposition, concept entanglement, destructive interference, greedy collapse by utility density.
- 🔗 **Automatic Entanglement** — worker automatically builds `entangled_with` and `supersedes_by` edges.
- ⏳ **Temporal Decay** — unaccessed memories decay over time (configurable half-life) and reactivate upon retrieval.
- 🗃️ **Logical Collections** — automatic routing by `type`/`subtype`: facts, preferences, rules, coding best practices, error resolutions, episodes.

**🛰️ Pipeline & Reliability**
- 📡 **Asynchronous Ingestion** — Redis Streams (consumer groups, ACK, bounded retries, dead-letter queue): zero latency for the calling agent.
- 🧩 **Structured LLM Extraction** — memory classification into validated JSON schemas, with regex fallback heuristics.
- 🪪 **Idempotency** — `idempotency_key`: submitting the same event twice results in a single consolidated memory.
- 🏊 **PostgreSQL Connection Pooling** & **HNSW Indexing** for ultra-fast vector search.

**🔌 Integrations**
- 🐍 **Python SDK** (`synaptiq-sdk`) & 🟦 **TypeScript SDK** (`@synaptiq/sdk`) out of the box.
- 🧰 **MCP Server** — native FastMCP tools (`store_memory`, `recall_memories`, `build_context`) for Claude Desktop, Cursor, and any MCP client.
- 🧬 **Pluggable Embeddings** — LM Studio (local) by default, OpenRouter, OpenAI, or NVIDIA NIM on demand.

**🔐 Security**
- 🏰 **Multi-Tenant Isolation** enforced server-side, Bearer API key authentication, rate limiting, CORS configuration, **GDPR purge endpoint**.

<br>

## 🏗️ Architecture

Modular async architecture: the agent never waits for heavy processing, all extraction and graph building is offloaded to background workers.

```mermaid
graph LR
    A[🤖 AI Agent] -->|1. events| API[⚙️ FastAPI Server]
    API -->|2. push job| R[(📨 Redis Streams)]
    R -->|3. pull job| W[🧠 Consolidation Worker]
    W -->|4. embedding + entanglement| DB[(🗄️ PostgreSQL + pgvector)]
    A -->|5. context/build| API
    API -->|6. Q-EM recall| DB
    DB -->|7. context packet| API
    API -->|8. rehydrated prompt| A
```

| Component | Role |
|---|---|
| **API** (`apps/api`) | Event ingestion (`/events`), context assembly (`/context/build`, `/retrieve`), direct memory writing (`/memories`). |
| **Worker** (`apps/worker`) | Consumes stream, classifies → embeds → entangles memories in graph. |
| **Core** (`packages/core`) | Shared logic: pluggable `Embedder`, governance, pure Q-EM mathematical engine. |
| **SDKs** (`packages/sdk-python`, `packages/sdk-typescript`) | Native client libraries for Python & TypeScript/JavaScript. |
| **MCP** (`apps/mcp`) | Exposed tools for AI agents via Model Context Protocol (`stdio` & `http`). |

<br>

## 🚀 Quickstart

### Prerequisites
- **Docker** & Docker Compose
- **LM Studio** (local) or an **OpenRouter / OpenAI API Key** (see [Embeddings](#-embeddings))
- Python **3.11+** (for non-container local development)

### Option A — Full Stack via Docker (Recommended)

```bash
git clone https://github.com/Jimmyjoe13/synaptiq.git
cd synaptiq
cp .env.example .env          # Adjust EMBEDDING_PROVIDER & API keys as needed
docker compose up --build     # Starts Postgres + Redis + API + Worker + MCP
```

Once `synaptiq-api` reports `healthy`:

```bash
curl http://127.0.0.1:8000/v1/health   # -> {"status":"ok", ...}
```

| Service | Endpoint |
|---|---|
| API (`v1`) | `http://127.0.0.1:8000` |
| MCP (SSE) | `http://127.0.0.1:8765` |
| PostgreSQL | `127.0.0.1:5435` |
| Redis | `127.0.0.1:6399` |

<details>
<summary><strong>Option B — Local Dev (Without Docker Containers for Python)</strong></summary>

<br>

```bash
# 1. Start database infrastructure only
docker compose up -d postgres redis

# 2. Install dependencies
pip install -r requirements-dev.txt
pip install -e packages/core -e packages/sdk-python

# 3. Start FastAPI server (port 8000)
python -m uvicorn apps.api.main:app --reload --port 8000

# 4. Start Consolidation Worker (separate terminal)
python apps/worker/worker.py
```

</details>

<br>

## 🌌 The Q-EM Engine in 4 Phases

A quantum-inspired metaphor, a deterministic algorithmic pipeline. On every `/v1/context/build`:

```
  1. SUPERPOSITION      Vector similarity search (pgvector) → scored memory candidates
        │               score = cosine_similarity × recency_factor
        ▼
  2. ENTANGLEMENT       Damped activation spreading along 'entangled_with' graph edges
        │               → related memories surface even if they don't match query keywords
        ▼
  3. INTERFERENCE       Destructive filtering:
        │                 • contradictions → obsolete versions are superseded
        ▼                 • redundancies (cosine > threshold) → only one survives
  4. MEASUREMENT        Greedy collapse by utility density per token,
                        routed into the 7 context packet collections
```

The algorithmic core lives in `packages/core/synaptiq_core/qem.py` — **pure functions, zero I/O, 100% unit tested**.

<br>

## 🐍 SDK Usage

### Python (`synaptiq-sdk`)

```python
from synaptiq_sdk import SynaptiqClient

client = SynaptiqClient("http://127.0.0.1:8000", api_key="your_api_key")

# 1. Capture raw agent interaction (async consolidation by worker)
client.capture(
    agent_id="george",
    session_id="sess_1",
    content="User prefers concise progress reports in English.",
)

# 2. Rehydrate compact context packet before calling LLM
ctx = client.build_context(
    agent_id="george",
    session_id="sess_1",
    task="Draft weekly status update",
    query="format and language preferences",
)

packet = ctx["context_packet"]                 # facts / preferences / rules / errors...
print(ctx["token_estimate"], packet["preferences"])
```

### TypeScript (`@synaptiq/sdk`)

```typescript
import { SynaptiqClient } from "@synaptiq/sdk";

const client = new SynaptiqClient({
  baseUrl: "http://127.0.0.1:8000",
  apiKey: "your_api_key",
});

await client.capture({
  agentId: "george",
  sessionId: "sess_1",
  content: "User prefers concise progress reports in English.",
});

const ctx = await client.buildContext({
  agentId: "george",
  sessionId: "sess_1",
  task: "Draft weekly status update",
  query: "format and language preferences",
});
```

<br>

## 📡 API Reference (`/v1`)

| Method | Endpoint | Role |
|:---:|---|---|
| `GET` | `/v1/health` | Service health status (Postgres + Redis) |
| `POST` | `/v1/events` | Async event ingestion (idempotent, queued to Redis Streams) |
| `POST` | `/v1/memories` | Direct write of pre-consolidated memory (returns the target `collection`) |
| `POST` | `/v1/retrieve` | Semantic vector search + Full-Text Search (FTS) |
| `POST` | `/v1/context/build` | Q-EM context packet assembly under token budget |
| `DELETE` | `/v1/memories` | GDPR purge — requires `admin` scope **and** `?confirm=<tenant_id>` (optional `?agent_id=` filter) |

<br>

## 🧰 Installing the MCP Server

The MCP server (`apps/mcp/server.py`) exposes three tools to any MCP client —
`store_memory`, `recall_memories`, `build_context`. It is a **thin HTTP client over the
SynaptiQ API**: it never touches PostgreSQL or Redis itself. That single fact dictates the
install order, and most failed setups are a violation of it:

```
  docker compose (Postgres + Redis)  →  API on :8000  →  MCP server  →  MCP client
```

If the API is not reachable, every tool call returns an `[ERROR]` string — the server itself
still starts and still lists its tools. Diagnose from the API outwards, never from the client
inwards.

### Configuration reference

| Variable | Required | Default | Role |
|---|:---:|---|---|
| `SYNAPTIQ_AGENT_ID` | **yes** | *none* | Memory identity this server reads and writes under. |
| `SYNAPTIQ_API_URL` | no | `http://127.0.0.1:8000` | Base URL of the SynaptiQ API. |
| `SYNAPTIQ_API_KEY` | for `http` | *empty* | Bearer key forwarded to the API. Mandatory when the transport is not `stdio`. |
| `MCP_TRANSPORT` | no | `stdio` | `stdio` or `http`. See below — this choice matters. |
| `MCP_HOST` / `MCP_PORT` | no | `0.0.0.0` / `8765` | Listen address in `http` transport. |
| `SYNAPTIQ_AUTOSTART_API` | no | `true` | Spawn `uvicorn` if the API does not answer. Set to `false` whenever a supervisor already owns the API. |

> [!IMPORTANT]
> **`SYNAPTIQ_AGENT_ID` has no default, on purpose.** It once defaulted to a fixed value, and
> a deployment whose memories had been written under a different identity read an empty
> partition and answered *"no memory found"* — no error, no warning. For a memory engine that
> symptom is indistinguishable from a genuinely empty store, so it is undebuggable from the
> outside.
>
> The server nevertheless **starts** without it and fails at *tool-call* time with a full
> explanation. That is also deliberate: an MCP server that exits on boot only shows the client
> `exit status 1`, stderr is discarded, and the server vanishes from the list. Failing fast is
> only worth it if somebody reads the failure.
>
> Find the identity your existing memories were written under:
> ```sql
> SELECT agent_id, count(*) FROM memories GROUP BY 1 ORDER BY 2 DESC;
> ```

### Choosing a transport

| | `stdio` | `http` |
|---|---|---|
| Process model | Client spawns a child process | Long-lived server, client just connects |
| Client entry | `command` + `args` | `serverUrl` |
| Needs `SYNAPTIQ_API_KEY` | no | **yes** |
| Best for | Claude Desktop, Cursor | antigravity CLI, containers, several clients sharing one server |

> [!WARNING]
> **Known limitation of `stdio` with antigravity CLI.** After stdin closes, `mcp.run()` takes
> 141–250 ms to unwind (measured; the time is spent in fastmcp's anyio loop, so an `os._exit()`
> changes nothing). antigravity CLI grants roughly a 100 ms grace window before calling
> `Kill()`; on Windows `TerminateProcess(handle, 1)` reads as `exit status 1`, and its manager
> then abandons the reload of **every** MCP server it owns. Node servers fit under that window,
> Python does not.
>
> With that client, use the `http` transport. There is no child process to stop, so there is no
> grace window to respect. The same change fixed the same symptom on the Obsidian MCP server.

<br>

### Reference install — Windows, self-hosted, antigravity CLI

This is a real, running deployment: one instance directory, Docker for storage only, Python
services in a local venv, LM Studio on the host for embeddings, and the MCP server exposed over
HTTP on `127.0.0.1:8765`.

<table>
<tr><td><strong>Instance dir</strong></td><td><code>C:\Users\jimmy\synaptiq</code> (a <code>git clone</code>, updated with <code>git pull</code> — never edited locally)</td></tr>
<tr><td><strong>Storage</strong></td><td>Docker: Postgres <code>127.0.0.1:5435</code>, Redis <code>127.0.0.1:6399</code></td></tr>
<tr><td><strong>Embeddings</strong></td><td>LM Studio on the host, <code>http://localhost:1234/v1</code>, 384-dim multilingual model</td></tr>
<tr><td><strong>API</strong></td><td><code>127.0.0.1:8000</code>, venv process</td></tr>
<tr><td><strong>MCP</strong></td><td><code>127.0.0.1:8765</code>, <code>http</code> transport, venv process</td></tr>
<tr><td><strong>Client</strong></td><td>antigravity CLI, <code>~/.gemini/antigravity-cli/mcp_config.json</code></td></tr>
</table>

**1 — Clone the instance and create the venv (Python 3.11)**

```powershell
git clone https://github.com/Jimmyjoe13/synaptiq.git C:\Users\jimmy\synaptiq
cd C:\Users\jimmy\synaptiq
py -3.11 -m venv .venv
# fastmcp MUST be installed in the SAME command as requirements.txt. Installed separately,
# the resolver bumps starlette past what fastapi==0.115.6 supports, and the API then dies
# with: TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'
.\.venv\Scripts\pip install -r requirements.txt fastmcp
```

**2 — Write the instance `.env`**

```env
DATABASE_URL=postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db
REDIS_URL=redis://127.0.0.1:6399/0

EMBEDDING_PROVIDER=lmstudio
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf
EMBEDDING_DIM=384

SYNAPTIQ_TENANT=default
SYNAPTIQ_AUTH_REQUIRED=true

# --- MCP server ---
SYNAPTIQ_API_URL=http://127.0.0.1:8000
SYNAPTIQ_AGENT_ID=antigravity_orchestrator
SYNAPTIQ_API_KEY=sk-synaptiq-...
MCP_TRANSPORT=http
MCP_HOST=127.0.0.1
MCP_PORT=8765
```

> [!CAUTION]
> **Keep this file pure ASCII on Windows.** `slowapi` re-reads it through
> `starlette.config.Config`, which opens it *without* specifying an encoding — so cp1252. A
> single non-representable UTF-8 byte (one emoji is enough: `0x8f`, from the U+FE0F variation
> selector) crashes the API at boot on an opaque `UnicodeDecodeError`, far from its cause.
>
> **Never change `EMBEDDING_MODEL` on a populated instance without re-embedding.** Two
> different 384-dim models raise no error at all — the stored vectors simply stop being
> comparable and recall degrades *in silence*. Verify with the cosine between a stored vector
> and the one recomputed by the current model: it must be `1.000`.

**3 — Create the API key for the MCP server**

Give it exactly what an agent needs — `read` + `write`, scoped to that one agent, and **no
`admin`**, so the GDPR purge stays out of the model's reach:

```powershell
.\.venv\Scripts\python scripts\create_api_key.py --name "mcp-antigravity" `
    --scopes read write --agents antigravity_orchestrator
```

Copy the printed key into `SYNAPTIQ_API_KEY`. It is stored only as a SHA-256 hash and is never
recoverable.

**4 — Start storage, then the services**

```powershell
docker compose up -d postgres redis migrate
.\scripts\start_services.ps1 -WaitForInfra 300
```

`start_services.ps1` is idempotent (a port already listening is left alone) and starts both the
API and the MCP server in `http` transport. `-WaitForInfra 300` is what makes it safe at logon:
Docker Desktop often needs one to two minutes to raise its containers, and an API started
before them keeps a NULL pool and answers `503` to everything, forever — it never recovers on
its own.

To run it at every logon, register it as a scheduled task:

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument '-NoProfile -WindowStyle Hidden -File "C:\Users\jimmy\synaptiq\scripts\start_services.ps1" -WaitForInfra 300'
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "SynaptiQ - services (instance)" -Action $action -Trigger $trigger
```

**5 — Declare the server in the client**

`~/.gemini/antigravity-cli/mcp_config.json` — with the `http` transport the whole entry is one
line, because there is no process for the client to manage:

```json
{
  "mcpServers": {
    "synaptiq": {
      "serverUrl": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

<details>
<summary><strong>Same install with the <code>stdio</code> transport (Claude Desktop, Cursor)</strong></summary>

<br>

Skip step 4's MCP process — the client spawns it. Set `MCP_TRANSPORT=stdio` and declare:

```json
{
  "mcpServers": {
    "synaptiq": {
      "command": "C:\\Users\\jimmy\\synaptiq\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\jimmy\\synaptiq\\apps\\mcp\\server.py"],
      "env": {
        "SYNAPTIQ_API_URL": "http://127.0.0.1:8000",
        "SYNAPTIQ_API_KEY": "sk-synaptiq-...",
        "SYNAPTIQ_AGENT_ID": "antigravity_orchestrator",
        "MCP_TRANSPORT": "stdio",
        "SYNAPTIQ_AUTOSTART_API": "false"
      }
    }
  }
}
```

Two rules here:

- Use the **absolute path to the venv's `python.exe`** and the **absolute path to
  `server.py`** — not `-m apps.mcp.server` with a `cwd`. The script fixes up `sys.path`
  itself and runs from any working directory; clients that ignore `cwd` would otherwise fail
  with `ModuleNotFoundError`, which surfaces only as an opaque `exit status 1`.
- Set `SYNAPTIQ_AUTOSTART_API=false` when a supervisor already owns the API, otherwise the
  MCP server will spawn a second `uvicorn` that loses the race on port 8000.

</details>

<br>

### Verifying the install

Run all three — each one clears a different failure mode, and the first two can pass while the
memory is still silently empty.

```powershell
# 1. Both services listening
.\scripts\start_services.ps1 -Status

# 2. The API is actually healthy (not just bound to the port)
curl http://127.0.0.1:8000/v1/health      # -> {"status":"ok","services":{...}}

# 3. THE ONE THAT MATTERS: the configured identity owns the memories
docker exec synaptiq-postgres psql -U synaptiq -d synaptiq_db `
  -c "SELECT agent_id, count(*) FROM memories GROUP BY 1 ORDER BY 2 DESC;"
```

Check that `SYNAPTIQ_AGENT_ID` appears in that list with a non-zero count. A mismatch is the
one failure the tools cannot report: `recall_memories` answers *"no matching memory found"*,
which reads exactly like a fresh install.

Then, from the client, ask the agent to call `recall_memories` on a subject you know is stored.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Client shows `exit status 1`, all MCP servers stop reloading | `stdio` shutdown exceeds the client's grace window | Switch to `MCP_TRANSPORT=http` and declare `serverUrl` |
| `[ERROR] ... SYNAPTIQ_AGENT_ID n'est pas defini` | No identity configured | Set `SYNAPTIQ_AGENT_ID` in the server env |
| *"No matching memory found"*, no error | Identity mismatch, **or** `EMBEDDING_MODEL` differs from the one that wrote the vectors | Check #3 above; verify the cosine of a stored vector against a recomputed one is `1.000` |
| `[ERROR] ... Connection refused` | API down, or `SYNAPTIQ_API_URL` wrong | `curl /v1/health`; check `api_error.log` in the instance root |
| `401 Clé API requise` on every tool | `SYNAPTIQ_AUTH_REQUIRED=true` and no `SYNAPTIQ_API_KEY` in the MCP env | Create a key (step 3) and set it |
| `403 Permission 'write' absente` | Key issued read-only | Re-issue with `--scopes read write` |
| API dies at boot on `UnicodeDecodeError` | Non-ASCII byte in `.env` on Windows | Rewrite `.env` in pure ASCII |
| `TypeError: Router.__init__() ... 'on_startup'` | `fastmcp` installed separately, starlette bumped | Reinstall: `pip install -r requirements.txt fastmcp` in one command |
| Tools work, `/events` never becomes a memory | The `relay` and `worker` services are not running | `docker compose up -d relay worker` — `/events` returns `201` and queues in the outbox even when nothing consumes it |

The MCP server logs to **stderr** (`mcp_error.log` when started by `start_services.ps1`), never
to stdout — in `stdio` transport stdout carries the JSON-RPC frames and a single log line would
corrupt the session.

<br>

## 🧬 Embedding Providers

Configurable via `EMBEDDING_PROVIDER` (`lmstudio` by default, `openrouter`, `openai`, `mock`).

```env
# Local LM Studio (default)
EMBEDDING_PROVIDER=lmstudio
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf
EMBEDDING_DIM=384

# OpenRouter (Cloud)
EMBEDDING_PROVIDER=openrouter
EMBEDDING_API_KEY=sk-or-v1-your_openrouter_key
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIM=1536
```

<br>

## 🔐 Security

SynaptiQ is **self-hosted: 1 deployment = 1 security boundary**. Tenant scoping is server-side (`SYNAPTIQ_TENANT`) and **never** trusted from client payloads. Agent separation within a tenant is isolated via `agent_id`.

```bash
# Generate a new Bearer API Key (default permissions: read + write)
python scripts/create_api_key.py --name "agent-prod"

# Read-only key
python scripts/create_api_key.py --name "dashboard" --scopes read

# Key restricted to a single agent's memory
python scripts/create_api_key.py --name "agent-A" --agents agentA

# Key allowed to run the GDPR purge (must be requested explicitly)
python scripts/create_api_key.py --name "ops" --scopes read write admin

# Include in HTTP requests: Authorization: Bearer <key>
```

`SYNAPTIQ_AUTH_REQUIRED=true` (default) ensures secure endpoints out of the box.

**Key scopes.** Every key carries permissions (`read`, `write`, `admin`) and an optional
agent whitelist. Two consequences worth knowing:

- **`agent_id` is enforced, not advisory.** A key created with `--agents agentA` gets `403`
  if it tries to read or write as `agentB`. Without `--agents`, the key reaches every agent
  of its tenant (historical behaviour, and the normal case for a single-agent instance).
- **The GDPR purge needs `admin` plus `?confirm=<tenant_id>`.** Keys issued before this
  change keep read and write, and lose the ability to wipe the instance.

The MCP server's agent identity comes from `SYNAPTIQ_AGENT_ID` in its environment — it is no
longer a tool parameter, so no prompt can make the model act as another agent.

**`SYNAPTIQ_AGENT_ID` is required and has no default.** Without it every tool call fails with
an actionable message. This is deliberate: it used to default to a fixed value, and a
deployment whose memories had been written under a different identity would read an empty
partition and answer *"no memory found"* — with no error. For a memory engine that symptom is
indistinguishable from a genuinely empty store, so it is undebuggable from the outside. See
[Installing the MCP Server](#-installing-the-mcp-server) for why the server still *boots*
without it rather than exiting. Find the identity of existing memories with:

```sql
SELECT agent_id, count(*) FROM memories GROUP BY 1;
```

<br>

## 🧪 Testing & CI

```bash
# Unit tests (no external services needed)
pytest tests/unit

# Full test suite (requires Postgres + Redis running)
docker compose up -d postgres redis
pytest tests/
```

CI workflows (`.github/workflows/ci.yml`) run `ruff` linting and test execution on every commit.

<br>

---

<div align="center">

**MIT License** — see [`LICENSE`](LICENSE).

<sub>Built for AI agents that cannot afford to forget. 🧠</sub>

</div>
