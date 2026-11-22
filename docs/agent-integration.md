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
| `entangled_with` edges | yes | yes — **since 0.3.1 only**, see below |
| Retry safety | per **event** (`idempotency_key`) | per **content** (`content_hash`) — §1.1 |
| Caller waits | no | yes, for one embedding |

Note that the two retry protections are not the same guarantee. `/v1/events` deduplicates a
replayed *event*; it does not stop two different events from yielding the same fact twice. The
benchmark corpus shows the difference plainly: `Melanie thanked Caroline.` exists 14 times,
from 14 distinct events. Content-level duplication is possible on both paths — they simply
protect against different failures.

Both paths build the graph. That is recent, and it matters for any instance created before
2026-08-01.

> [!IMPORTANT]
> **Before this fix, an agent that wrote only through `store_memory` built no graph at all.**
>
> `_entangle()` lived in `apps/worker/worker.py` and was called from nowhere else. The API read
> `relationships` for the entanglement phase but never wrote a row to it. So phase 2 of Q-EM —
> activation spreading, the mechanism that surfaces a related memory the query never mentioned
> — had nothing to spread through, and recall silently degraded to phase 1 + phase 3. Measured
> on a real instance: **28 memories, 0 edges**, after weeks of use.
>
> Nothing reported it, and **nothing fills it in retroactively** — entanglement is a
> write-time effect. If your instance predates the fix, rebuild once.

Check any instance directly:

```sql
SELECT m.agent_id, count(DISTINCT m.id) AS memories, count(r.source_memory_id) AS edges
FROM memories m
LEFT JOIN relationships r
       ON r.source_memory_id = m.id AND r.relation_type = 'entangled_with'
WHERE m.status = 'active'
GROUP BY 1 ORDER BY 2 DESC;
```

Or read `synaptiq_graph_edges_per_memory{agent_id="..."}` from `/metrics` — that gauge exists
so an empty graph stops being invisible.

**To fill an existing graph**, or after changing the threshold:

```bash
python scripts/rebuild_entanglement.py --agent my_agent --dry-run   # state + distribution
python scripts/rebuild_entanglement.py --agent my_agent
```

`--dry-run` prints the nearest-neighbour similarity histogram with your current threshold
marked, which is the only honest way to choose one (§8). The script is idempotent, only weaves
the compartments that entangle, and emits **only** `entangled_with`: the write path also creates
a `supersedes_by` edge between `savoir_faire` and `alertes` memories, and
replaying *that* in bulk would make the interference phase cancel memories that are still
valid. Suppression stays a deliberate write, never a side effect of maintenance.

One thing the graph does **not** do: `propagate_entanglement` only activates a link when *both*
endpoints are already in the candidate set. The graph therefore **re-ranks the candidate pool,
it does not widen it** — it promotes a memory that matched weakly, it cannot retrieve one that
hybrid search missed entirely. Worth knowing before attributing a recall miss to the graph.

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

## 2. Contradiction handling only covers `reflexes`

`handle_contradictions` returns immediately unless the memory is in the `reflexes`
compartment (`packages/core/synaptiq_core/governance.py`):

```python
if new_memory.get("compartment") != "reflexes":
    return []
```

So nothing is ever archived automatically in `savoirs`, `decisions`, `environnement` or any
other compartment — **even with a live LLM judge**. And the judge itself is fail-closed:
`CONTRADICTION_JUDGE=auto` with no LLM configured resolves to `no_judge`, which archives
nothing at all.

The practical consequence for an agent author: **keeping the memory true over time is your
job, in every compartment but one.** Write a correction and you now have two contradictory
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

## 3. File in the right compartment

Since v0.4 there is nothing to design: the memory has **nine fixed compartments**, modelled
on regions of the brain, each with its own behaviour (see the README). The agent's only job is
to pick the right one — and if it does not, the engine classifies by similarity and says so.

For a coding/ops agent, the usual mapping:

| Compartment | What earns a place in it |
|---|---|
| `reflexes` | How this user wants the agent to work; rules to apply every time (injected into **every** context) |
| `decisions` | A technical choice, **its reason, and the options rejected** |
| `environnement` | Hosts, ports, paths, where the credentials live |
| `savoir_faire` | Command sequences proven on the real target, coding conventions |
| `alertes` | An error and its fix; a trap not to fall into again |
| `savoirs` | What is deployed, what is blocked, what is left to do |
| `episodes` | End-of-session journal: done / shipped / tested / left open |
| `observations` | A measurement or audit result, true at one point in time |

Keep `reflexes` short and few: they are injected regardless of the question (up to
`QEM_REFLEXES_INJECTES`, at most `QEM_REFLEXES_MAX_SHARE` of the budget). A project fact filed
there costs tokens in every unrelated session.

Read the `compartment_origin` field of the write response: `moteur` means the engine chose for
the agent, `alias` that an old v0.3 name was converted. An agent that ignores it will believe
in a filing it did not make.

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
| same query, `compartments=["decisions"]` | the deployment policy decision, first |

`build_context` is the rehydration tool: it runs the full four-phase pipeline, routes results
into packet sections, drops redundancies, and respects a token budget. Use it to prepare a
turn; use `recall_memories` to answer a narrow lookup.

**And filter by compartment as soon as you know it.** It is the highest-leverage
parameter on both tools — fewer candidates means less noise at the same token budget.

> [!NOTE]
> `context_packet` has a **fixed shape**: the nine compartments, always present, in order,
> even when empty. `reflexes` is filled even when nothing matched the query.

---

## 6. Do not leave the protocol to the model's judgment

The usual integration is three instructions in a system prompt: rehydrate at the start, write
durable facts as they appear, journal at the end. Instructions get skipped — reliably, and
exactly when the request is urgent and the context is full.

Anything you actually depend on belongs in the harness, not the prompt. With a hook-capable
client (the example below is Claude Code, the shape generalises):

**Session start — inject the always-true slice, narrowly.** At startup the task is unknown, so
a full `build_context` is a guess: a large packet, mostly off-topic, and worse than wasted
tokens because it primes the agent toward the wrong project. Inject only the compartment that
holds in *every* session (`reflexes`) — how the user wants to work, their format preferences — and leave a
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
near ~1 edge per memory. The rebuild script prints exactly that histogram, with your current
threshold marked:

```bash
python scripts/rebuild_entanglement.py --agent my_agent --dry-run
```

Too low is not free either: spurious edges spread activation into unrelated memories and the
packet fills with plausible noise.

> [!WARNING]
> **Changing the threshold does not touch memories already written.** It applies to subsequent
> writes only, so lowering it and observing no improvement is the expected outcome, not a
> refutation. Re-run `rebuild_entanglement.py` (without `--dry-run`) after any change —
> and with `--purge` if you *raised* it and want the graph actually tightened, knowing that
> purge also removes edges legitimately laid down at write time.

The default is deliberately left at `0.7`. Lowering it in the shipped configuration would
change recall for every existing deployment, silently — the same class of change this project
has been fixing elsewhere.

---

## 9. Verification checklist

Run all of it. The first three can pass while recall is quietly broken.

| # | Check | Passing looks like |
|---|---|---|
| 1 | `curl /v1/health` | `postgres`, `redis` **and** `ingestion` all healthy |
| 2 | MCP tool call, not just server start | a real `store_memory` returning its resolved `compartment` |
| 3 | `SELECT agent_id, count(*) FROM memories GROUP BY 1` | your `SYNAPTIQ_AGENT_ID`, non-zero |
| 4 | **Edge count for that agent** — `synaptiq_graph_edges_per_memory` or the query in §1 | non-zero, roughly ~1 per memory |
| 5 | Embedding coherence | cosine of a stored vector vs. one recomputed now = `1.000` |
| 6 | **A real `build_context`** on a task you know is covered | the right packet, inside budget |

Check 6 is the only one that tests the thing you actually built. Ask for something you know is
stored, and read what comes back.

---

## Related

- [`mcp-server.md`](mcp-server.md) — transports, install, client configuration, troubleshooting
- [`configuration.md`](configuration.md) — every environment variable
- [`../README.md`](../README.md) — architecture, Q-EM phases, compartments
