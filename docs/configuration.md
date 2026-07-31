# Configuration reference

Every setting is an environment variable, read from the **repository-root `.env`** (single
source — the API, worker, relay and MCP server all load that file explicitly).
[`.env.example`](../.env.example) is the annotated template.

Settings are read **at call time**, not frozen at import, so a benchmark or a test can vary
one phase without a redeploy.

---

## Storage

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://synaptiq:synaptiq_password@127.0.0.1:5435/synaptiq_db` | Compose overrides it to the internal hostname for containers. |
| `REDIS_URL` | `redis://127.0.0.1:6399/0` | The DB index separates environments on a shared Redis. |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `10` | API connection pool. |
| `WORKER_DB_POOL_MIN` / `WORKER_DB_POOL_MAX` | `1` / `4` | Worker pool. |

## Embeddings

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `lmstudio` | `lmstudio` · `openrouter` · `openai` · `mock` (**tests only** — non-semantic vectors). |
| `EMBEDDING_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint. |
| `EMBEDDING_MODEL` | `text-embedding-paraphrase-multilingual-minilm-l12-v2.gguf` | |
| `EMBEDDING_DIM` | `384` | **Must match the `VECTOR(n)` column.** Changing it is a schema migration. |
| `EMBEDDING_API_KEY` | — | Also read from `OPENROUTER_API_KEY` / `OPENAI_API_KEY`. |
| `EMBEDDING_COHERENCE_CHECK` | `true` | Worker refuses to start if the model disagrees with the stored vectors. |
| `EMBEDDING_COHERENCE_MIN` | `0.999` | Cosine floor for that check. |

> [!CAUTION]
> **`EMBEDDING_DIM` protects only against the noisy case.** The dangerous one is silent: two
> *different* models of the *same* dimension. No exception, no log — the vectors simply stop
> being comparable and recall degrades invisibly. That is what
> `EMBEDDING_COHERENCE_CHECK` exists for: at startup the worker re-embeds a stored memory and
> compares. Disable it only for a deliberate model migration, with a full re-embed.

## LLM extraction (worker)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` = regex heuristics. See the warning below. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | A local endpoint needs no API key. |
| `LLM_MODEL` | `meta-llama/llama-3-8b-instruct:free` | |
| `LLM_API_KEY` | — | |
| `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF_S` | `3` / `2.0` | Honours `Retry-After`; `429` is frequent on free tiers. |
| `MAX_FACTS_PER_EVENT` | `5` | A dialogue turn rarely states more. |

> [!WARNING]
> **`LLM_PROVIDER=mock` caps what Q-EM can demonstrate.** Regex heuristics mostly produce
> `episodic/interaction`, and episodes are excluded from entanglement by default — so the
> graph stays empty and Q-EM degenerates into plain top-k. This only affects the `/events`
> path; `POST /v1/memories` (and the MCP `store_memory`) are already structured by the caller.

## Retrieval

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_HYBRID` | `true` | `false` = pure vector ranking. Useful to measure the hybrid's contribution. |
| `RETRIEVAL_CANDIDATES` | `50` | Candidates **per path** before fusion. |
| `RRF_K` | `60` | Reciprocal-rank-fusion damping; 60 is the reference value. |
| `RRF_WEIGHT_VECTOR` / `RRF_WEIGHT_FTS` | `1.0` / `1.0` | Relative weight of each path. |

Fusion is by **rank**, not by score: cosine similarity and `ts_rank` do not live on the same
scale, so combining their raw values would be meaningless.

## Q-EM engine

| Variable | Default | Notes |
|---|---|---|
| `QEM_ENTANGLE_THRESHOLD` | `0.7` | Cosine above which the worker links two memories. |
| `QEM_ENTANGLE_DAMPING` | `0.5` | Activation attenuation per hop. |
| `QEM_ENTANGLE_MAX_HOPS` | `2` | `1` = historical single-hop behaviour; `0` disables spreading. |
| `QEM_ENTANGLE_TYPES` | `procedural,semantic` | Legacy instance-wide fallback — superseded by each collection's `entangle` flag. |
| `QEM_REDUNDANCY_THRESHOLD` | `0.75` | Above this cosine, only the higher-priority candidate survives. |
| `QEM_RECENCY_HALFLIFE_DAYS` | `90` | Decay half-life since last access. `0` disables decay. |

## Collections

| Variable | Default | Notes |
|---|---|---|
| `MAX_COLLECTIONS_PER_AGENT` | `50` | A readability guardrail, not a technical limit. |
| `COLLECTION_DUP_THRESHOLD` | `0.85` | Cosine between *descriptions* above which creation is refused. |
| `COLLECTION_STALE_DAYS` | `14` | Declared but still empty after this → flagged `stale`. |

Deliberately higher than the memory redundancy threshold: wrongly refusing a legitimate
collection leaves the agent with nowhere to file, whereas a near-duplicate that slips through
can still be merged afterwards.

## Ingestion pipeline

| Variable | Default | Notes |
|---|---|---|
| `EVENT_STREAM` / `EVENT_GROUP` / `EVENT_DLQ` | `synaptiq:events` / `synaptiq-workers` / `synaptiq:events:dlq` | |
| `EVENT_MAX_DELIVERIES` | `5` | Redeliveries before the dead-letter queue. |
| `EVENT_IDLE_RECLAIM_MS` | `30000` | Idle time before a pending message is reclaimed. |
| `EVENT_RECLAIM_INTERVAL_MS` | `15000` | Forced reclaim interval, so retries progress even under constant load. |
| `OUTBOX_POLL_SECONDS` | `0.5` | Relay polling interval. |
| `IDEMPOTENCY_TTL` | `86400` | |
| `HEALTH_OUTBOX_MAX_AGE_S` | `300` | Age of the oldest unpublished event above which `/v1/health` reports `"ingestion":"stalled"`. |

## Security

| Variable | Default | Notes |
|---|---|---|
| `SYNAPTIQ_TENANT` | `default` | The instance's tenant. **Never accepted from a request payload.** |
| `SYNAPTIQ_AUTH_REQUIRED` | `true` | `false` allows unauthenticated `read`/`write`. `admin` is refused either way. |
| `AUTH_CACHE_TTL` | `60` | API-key cache. Bounded revocation delay; `0` disables the cache. |
| `AUTH_CACHE_MAX` | `1024` | |
| `RATE_LIMIT` | `120/minute` | Per client IP (slowapi). |
| `CORS_ORIGINS` | *(empty)* | No browser origin allowed by default — SynaptiQ is called server-to-server. |

> [!IMPORTANT]
> `SYNAPTIQ_AUTH_REQUIRED=false` is a convenience for a trusted localhost instance. It does
> **not** open the GDPR purge: `admin` always requires a key. A convenience flag must not
> unlock an irreversible endpoint.

## Contradictions

| Variable | Default | Notes |
|---|---|---|
| `CONTRADICTION_JUDGE` | `auto` | `auto` · `llm` · `off`. `auto` = no judge while `LLM_PROVIDER=mock`, hence no automatic archiving. |
| `CONTRADICTION_SIM_THRESHOLD` | `0.8` | **Pre-filter only** — it bounds how many candidates reach the judge. |

Similarity alone never archives anything. An earlier version treated "close" as
"contradictory" and silently destroyed perfectly compatible preferences ("short emails" vs.
"emails in French" sit around 0.85 cosine). Archiving now requires an explicit verdict, and
the judge is fail-closed: no judge configured → nothing is archived.

## Observability

| Variable | Default | Notes |
|---|---|---|
| `LOG_FORMAT` | `text` | `json` for structured logs. |
| `LOG_LEVEL` | `INFO` | |

## MCP server

See [`mcp-server.md`](mcp-server.md#2-configuration).
