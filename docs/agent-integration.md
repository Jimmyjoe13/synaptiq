# Integrating an agent

Standing the engine up is the easy half. Getting an agent to *use* it well is a separate
design problem, and it is where a memory deployment quietly fails: every call returns `200`,
every tool reports success, and the recall is mediocre for reasons nothing surfaces.

This guide is the part the [API reference](../README.md#api-reference) and the
[configuration reference](configuration.md) do not cover — what an agent author has to decide,
and the four asymmetries in the engine that will bite before anything logs a warning.

Read [`mcp-server.md`](mcp-server.md) first if you have not installed the server yet.

---

## 1. The two write paths are not equivalent

Both endpoints create memories. They do **not** create the same thing.

| | `POST /v1/events` | `POST /v1/memories` (and MCP `store_memory`) |
|---|---|---|
| Who processes it | the worker, asynchronously | the API, synchronously |
| Embedding | worker | API, before the insert |
| LLM extraction | yes — content is split into facts | none, you wrote the memory yourself |
| Contradictions | yes, **`preference` only** (§2) | yes, **`preference` only** (§2) |
| `entangled_with` edges | **yes** | **no** |
| Retry safety | per **event** (`idempotency_key`) | per **content** (`content_hash`) — §1.1 |
| Caller waits | no | yes, for one embedding |

Note that the two retry protections are not the same guarantee. `/v1/events` deduplicates a
replayed *event*; it does not stop two different events from yielding the same fact twice. The
benchmark corpus shows the difference plainly: `Melanie thanked Caroline.` exists 14 times,
from 14 distinct events. Content-level duplication is possible on both paths — they simply
protect against different failures.

> [!IMPORTANT]
> **An agent that writes only through `store_memory` builds no graph at all.**
>
> `_entangle()` lives in `apps/worker/worker.py` and is called from nowhere else. The API
> reads `relationships` for the entanglement phase (`apps/api/main.py`, `fetch_relationships`)
> but never writes a row to it. So phase 2 of Q-EM — activation spreading, the mechanism that
> surfaces a related memory the query never mentioned — has nothing to spread through.
>
> Recall silently degrades to phase 1 + phase 3: hybrid search plus interference. That is
> still a working memory, and it is *not* what the benchmark measured.

Nothing reports this. Check it directly:

```sql
SELECT m.agent_id, count(DISTINCT m.id) AS memories, count(r.source_memory_id) AS edges
FROM memories m
LEFT JOIN relationships r ON r.source_memory_id = m.id
WHERE m.status = 'active'
GROUP BY 1 ORDER BY 2 DESC;
```

A row with hundreds of memories and `0` edges is an agent whose graph was never built.

**Three ways out**, in decreasing order of purity:

1. **Write through `/v1/events`.** You get the full pipeline. The cost is that an LLM
   re-extracts your content: if the agent already produced a clean atomic memory, extraction
   can only degrade it, and with `LLM_PROVIDER=mock` the regex fallback will mangle it.
   Sensible for raw conversation capture, wrong for deliberate agent writes.
2. **Write directly and re-entangle periodically.** Keep `store_memory` for its precision and
   run a maintenance pass that replays the worker's logic. Idempotent, so schedule it on
   session end or on a timer:

   ```sql
   INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
   SELECT s.id, n.id, 'entangled_with', n.sim
   FROM memories s
   CROSS JOIN LATERAL (
       SELECT m.id, (1 - (m.embedding <=> s.embedding)) AS sim
       FROM memories m
       WHERE m.tenant_id = s.tenant_id AND m.agent_id = s.agent_id
         AND m.id <> s.id AND m.status = 'active'
       ORDER BY m.embedding <=> s.embedding
       LIMIT 3
   ) n
   WHERE s.agent_id = :agent AND s.status = 'active' AND n.sim > :threshold
   ON CONFLICT (source_memory_id, target_memory_id) DO NOTHING;
   ```

   Mirror the worker: top-3 nearest neighbours, threshold above `QEM_ENTANGLE_THRESHOLD`
   (see §8 before trusting the default). Emit **only** `entangled_with` — the worker also
   creates a `supersedes_by` edge between `coding_best_practices` and
   `code_error_resolution`, and replaying *that* in bulk makes the interference phase cancel
   memories that are still valid. Suppression must stay a deliberate write, never a
   side effect of a maintenance job.
3. **Accept single-hop recall.** Legitimate if your memory is a flat set of unrelated facts.
   Decide it, do not discover it.

### 1.1 Retries are safe, and that is recent

`POST /v1/memories` is **idempotent on content**. Writing the same text twice under the same
identity returns the same `memory_id`, the second time with `status: "duplicate"` (still
`201` — the same convention `/v1/events` uses for its own no-op). Deduplication is on the
normalised content, so casing and whitespace differences do not create a second memory.

Two things follow for an agent author:

- **Read the `status` field.** A `duplicate` is not a failure and not a creation. An agent
  that reads it as success may believe it added something it did not; one that reads it as
  failure may loop, rewording to "fix" a problem that does not exist. To correct a memory,
  write a *new* one stating the updated version and archive the old (§2) — re-sending the
  same text changes nothing by design.
- **`idempotency_key` is optional and secondary.** Supply it only if you have a genuinely
  stable key (a source row id in an import). A key generated per attempt protects nothing,
  since the retry generates a new one — which is exactly why content is the primary
  mechanism.

Before this existed, a client-perceived timeout on a call that had actually succeeded server
side left a permanent second row. The recall cost was small (phase 3 cancels the redundancy);
the graph cost was not, and it was permanent — see §1.

---

## 2. Contradiction handling only covers `preference`

`handle_contradictions` returns immediately unless the memory is exactly `semantic` +
`preference` (`packages/core/synaptiq_core/governance.py`):

```python
if new_memory.get("type") != "semantic" or new_memory.get("subtype") != "preference":
    return []
```

So nothing is ever archived automatically in `fact`, `rule`, `decisions`, `project_state`, or
any shelf you create — **even with a live LLM judge**. And the judge itself is fail-closed:
`CONTRADICTION_JUDGE=auto` with no LLM configured resolves to `no_judge`, which archives
nothing at all.

The practical consequence for an agent author: **keeping the memory true over time is your
job, in every collection but one.** Write a correction and you now have two contradictory
memories coexisting, both `active`, both recallable, with nothing to arbitrate between them.
For a memory engine that is the worst failure mode — the agent reads a stale fact and has no
way to know it is stale.

Superseding by hand, correctly:

```python
new_id = store_memory(corrected_content, ...)      # a real embedding, computed by the API
# UPDATE memories SET status = 'archived' WHERE id = old_id;
# INSERT INTO relationships (source_memory_id, target_memory_id, relation_type, weight)
#   VALUES (new_id, old_id, 'supersedes_by', 1.0);
```

The `supersedes_by` edge is what makes an archive distinguishable from a disappearance —
`qem.apply_contradictions` reads the same relation type at recall time.

> [!CAUTION]
> **Never fix a memory with `UPDATE memories SET content = ...`.** The embedding is not
> recomputed, so the stored vector silently stops describing the stored text. The memory
> still recalls — for the *old* wording. This is the same class of failure as changing
> `EMBEDDING_MODEL` on a populated instance: no error, no warning, degraded retrieval.
> Write a new memory and archive the old one.

---

## 3. Design the shelves before the first write

A `collection` is where a memory lands in the context packet. Deciding them up front is worth
more than any threshold tuning, because a memory filed nowhere useful is a memory that never
comes back at the right moment.

The seven system collections cover general assistants. An agent with a job needs its own. A
coding/ops agent, as a worked example:

| Collection | Family | What earns a place in it |
|---|---|---|
| `project_state` | `semantic` | What is deployed, what is blocked, what is left to do |
| `decisions` | `semantic` | A technical choice, **its reason, and the options rejected** |
| `infra_access` | `semantic` | Hosts, ports, paths, where the credentials live |
| `deploy_playbooks` | `procedural` | Command sequences proven on the real target |
| `session_log` | `episodic` | End-of-session journal: done / shipped / tested / left open |
| `user_workflow` | `procedural` | How this user wants the agent to work |

Four guardrails apply, and they are there because an unsupervised model creates one category
per nuance:

- **Semantic duplicate check** at `COLLECTION_DUP_THRESHOLD` (0.85) on the *description* —
  creation is refused and the near collection is named. Write descriptions that state a
  distinct purpose, not a synonym of another shelf.
- **Cap** `MAX_COLLECTIONS_PER_AGENT` (50), surfaced in `list_collections`.
- **Merge** via `POST /v1/collections/merge` — memories are relabelled, never destroyed.
- **Dormant shelves** flagged `stale` after `COLLECTION_STALE_DAYS` (14) with no writes.

Set `entangle` deliberately per collection: it decides whether that shelf feeds the multi-hop
path. Structured episodic content (meeting notes, session journals) is worth weaving in; raw
interaction logs are not, and `interaction` and `scratch` ship with `entangle=False` for that
reason.

Instruct the agent to call `list_collections` **before** inventing a `subtype`. An undeclared
name is accepted and routed to the family's fallback section — the response says so
explicitly (`collection`, `canonical_subtype`), and an agent that does not read that field
will believe in a filing that never happened.

---

## 4. Write memories that survive being recalled alone

Every memory is retrieved **on its own**, stripped of the conversation that produced it. That
single fact drives the whole writing style.

| Rule | Why |
|---|---|
| One fact per memory | A memory bundling five facts is recalled for one and spends tokens on four |
| Self-contained | No "as decided above", no "the server" — name it, every time |
| Absolute dates | "last Tuesday" is meaningless three months later; write `2026-03-24` |
| Carry the *why* | A decision without its reason gets re-litigated on every session |
| Include the counter-evidence | "X was rejected because Y" stops the agent re-proposing X |

Bad — a chat message, not a memory:

> Ok so as we discussed, I've switched the scraper to the other search backend since the
> previous one was rate-limiting us.

Good — recallable in six months, by an agent with no context:

> On 2026-03-29 DuckDuckGo was replaced by SearXNG in `url_finder.py`, after DDG returned
> zero results across an entire seed run on 2026-03-26 (total rate limiting). SearXNG is
> open source, needs no API key, and is used across 6 public instances in rotation with
> retry on 429 and timeout.

This matters more with a compressed similarity space (§8): short, distinct memories are both
cheaper to recall and less likely to be mistaken for each other.

---

## 5. Read with `build_context`, not raw `recall_memories`

`recall_memories` is a search tool: it returns the top *k* by hybrid relevance, and a dense,
memory-heavy project will dominate a generic query. Measured on a 57-memory store where one
project accounted for roughly half the corpus:

| Call | Top 3 |
|---|---|
| `recall_memories("deployment policy for the VPS", limit=3)` | three memories from an unrelated scraper project |
| same query, `collections=["decisions"]` | the deployment policy decision, first |

`build_context` is the rehydration tool: it runs the full four-phase pipeline, routes results
into packet sections, drops redundancies, and respects a token budget. Use it to prepare a
turn; use `recall_memories` to answer a narrow lookup.

**And filter by collection as soon as you know the shelf.** It is the highest-leverage
parameter on both tools — fewer candidates means less noise at the same token budget.

> [!NOTE]
> `context_packet` has **no fixed number of keys**. The seven canonical sections are always
> present and every declared collection adds one, even when empty. Iterate over the packet's
> entries; never read hardcoded keys.

---

## 6. Do not leave the protocol to the model's judgment

The usual integration is three instructions in a system prompt: rehydrate at the start, write
durable facts as they appear, journal at the end. Instructions get skipped — reliably, and
exactly when the request is urgent and the context is full.

Anything you actually depend on belongs in the harness, not the prompt. With a hook-capable
client (the example below is Claude Code, the shape generalises):

**Session start — inject the always-true slice, narrowly.** At startup the task is unknown, so
a full `build_context` is a guess: a large packet, mostly off-topic, and worse than wasted
tokens because it primes the agent toward the wrong project. Inject only the collections that
hold in *every* session — how the user wants to work, their format preferences — and leave a
line reminding the agent to call `build_context` once the task is known.

Read those rows straight from Postgres rather than through `/v1/context/build`: the API path
needs to embed a query, so it needs the embedding backend reachable *and* warm — measured
above 5 s cold, on a path where a session must never wait. A handful of rows needs no ranking
anyway. Degrade silently: on any failure write nothing and exit `0`. A startup hook that
prints a stack trace costs more than the context it delivers.

**Session end — run the graph maintenance from §1.** Idempotent, silent, and it must never
block shutdown (`|| true`).

That split leaves exactly one thing to the model's judgment — calling `build_context` once it
knows the task — because that is the only part a hook cannot do.

---

## 7. Identity is server-side, and it is the whole security model

`SYNAPTIQ_AGENT_ID` is server configuration and no tool takes it as a parameter. If it were a
parameter, the model would choose whose memory it reads and writes.

- **It has no default, deliberately.** A wrong identity reads an empty partition and answers
  "no memory found" — with no error. That symptom is indistinguishable from a fresh install,
  so it is undebuggable from the outside.
- **Scope the key to the agent and omit `admin`**:
  `create_api_key.py --scopes read write --agents my_agent`. The GDPR purge then stays out of
  the model's reach entirely, whatever a prompt asks for.
- **One identity is one brain, across every client and directory.** Declaring the MCP server
  at user scope rather than per-project gives an agent the same memory in every workspace —
  usually what you want, and worth being deliberate about, since two clients sharing an
  `agent_id` also share every write.

Verify the identity owns the memories before believing any recall result:

```sql
SELECT agent_id, count(*) FROM memories GROUP BY 1 ORDER BY 2 DESC;
```

---

## 8. Retune `QEM_ENTANGLE_THRESHOLD` for your language and model

The `0.7` default is calibrated on English corpora. It is not portable.

Field measurement — 55 short, deliberately non-redundant French memories,
`paraphrase-multilingual-MiniLM-L12-v2` (384 dims):

| Threshold | Edges built | Edges per memory |
|---|---|---|
| `0.70` (default) | 8 | 0.15 |
| `0.62` | 52 | 0.95 |

Nearest-neighbour similarities peaked between 0.50 and 0.68 — the model's similarity range is
compressed relative to the English-calibrated default, so almost nothing cleared 0.70 and the
graph stayed effectively empty. For reference, the LOCOMO benchmark corpus (English, 990
memories) built 1420 edges, about 1.43 per memory.

This is one corpus with one model, not a benchmark. The transferable part is the method:
before trusting the default, look at your own distribution and pick a threshold that lands
near ~1 edge per memory.

```sql
SELECT round(n.sim::numeric, 2) AS sim, count(*)
FROM memories s
CROSS JOIN LATERAL (
    SELECT (1 - (m.embedding <=> s.embedding)) AS sim FROM memories m
    WHERE m.agent_id = s.agent_id AND m.id <> s.id AND m.status = 'active'
    ORDER BY m.embedding <=> s.embedding LIMIT 3
) n
WHERE s.agent_id = :agent
GROUP BY 1 ORDER BY 1 DESC;
```

Too low is not free either: spurious edges spread activation into unrelated memories and the
packet fills with plausible noise.

---

## 9. Verification checklist

Run all of it. The first three can pass while recall is quietly broken.

| # | Check | Passing looks like |
|---|---|---|
| 1 | `curl /v1/health` | `postgres`, `redis` **and** `ingestion` all healthy |
| 2 | MCP tool call, not just server start | a real `store_memory` returning its resolved `collection` |
| 3 | `SELECT agent_id, count(*) FROM memories GROUP BY 1` | your `SYNAPTIQ_AGENT_ID`, non-zero |
| 4 | **Edge count for that agent** (§1) | non-zero, roughly ~1 per memory |
| 5 | Embedding coherence | cosine of a stored vector vs. one recomputed now = `1.000` |
| 6 | **A real `build_context`** on a task you know is covered | the right packet, inside budget |

Check 6 is the only one that tests the thing you actually built. Ask for something you know is
stored, and read what comes back.

---

## Related

- [`mcp-server.md`](mcp-server.md) — transports, install, client configuration, troubleshooting
- [`configuration.md`](configuration.md) — every environment variable
- [`../README.md`](../README.md) — architecture, Q-EM phases, families and collections
