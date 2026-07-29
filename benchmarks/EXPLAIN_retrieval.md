# Why the hybrid retrieval query does not factor out its filter

Reproduce with `python scripts/explain_retrieval.py --seed 5000` (fixed seed, dedicated
`__explain_bench__` tenant, `ANALYZE` run before measuring).

## The problem

Until 29/07 both hybrid queries factored the tenant/agent filter into a shared CTE:

```sql
WITH filtre AS (SELECT * FROM memories WHERE tenant_id = ... AND agent_id = ...),
     vectoriel   AS (SELECT id, ... FROM filtre ORDER BY embedding <=> $1 LIMIT 50),
     plein_texte AS (SELECT id, ... FROM filtre WHERE content_tsv @@ ...)
SELECT ... FROM filtre f LEFT JOIN vectoriel ... LEFT JOIN plein_texte ...
```

`filtre` is referenced **three times**. PostgreSQL only inlines a CTE when it is referenced
**once**; beyond that it is materialised. The `ORDER BY embedding <=> $1` therefore sorts a
materialised intermediate result rather than the table, and `idx_memories_embedding_hnsw`
becomes unusable.

## Measured — 5 000 memories, 384 dims, pgvector/pgvector:pg16

| | Plan on `memories` | HNSW index | Execution time (2 runs) |
|---|---|---|---|
| **Before** | `Seq Scan` (5 000 rows) → `CTE Scan on filtre` sorted | **not used** | 64.3 ms · 124.9 ms |
| **After** | `Index Scan using idx_memories_embedding_hnsw` | used | 6.2 ms · 9.3 ms |

Roughly **10× faster**. Absolute numbers move between runs (a developer laptop under
Docker Desktop); the plan shapes do not, and the plan shape is the point.

The important part is not the constant factor: the old plan is **linear in corpus size**
because every active memory of the agent gets an exact distance computation. The new plan is
sub-linear (approximate nearest neighbour). At 5 000 memories the gap is 10×; it widens with
every memory added, which is precisely the direction a long-term memory engine moves in.

## The fix

Each path queries `memories` directly and **repeats** the tenant/agent filter. The two rank
lists are then merged in a small `retenus` CTE — referenced once, so its materialisation is
irrelevant.

```sql
WITH vectoriel AS (
    SELECT id, row_number() OVER (ORDER BY distance) AS rank_vec
    FROM (SELECT id, embedding <=> $1::vector AS distance
          FROM memories WHERE tenant_id = $2 AND agent_id = $3 AND ...
          ORDER BY embedding <=> $1::vector LIMIT $4) v
), plein_texte AS ( ... same shape, ts_rank ... ),
retenus AS (
    SELECT id, min(rank_vec) AS rank_vec, min(rank_fts) AS rank_fts
    FROM (SELECT id, rank_vec, NULL::bigint FROM vectoriel
          UNION ALL
          SELECT id, NULL::bigint, rank_fts FROM plein_texte) u
    GROUP BY id
)
SELECT ... FROM retenus r JOIN memories m ON m.id = r.id;
```

**Do not "clean this up".** Factoring the repeated filter back into a shared CTE reverts the
regression, silently: the query keeps returning correct results, only ~10× slower and with a
cost that grows linearly. That is why the duplication carries a comment at both call sites.

## The other hot spot: redundancy filtering in Python

`filter_redundancy` (phase 3 of Q-EM) compared candidate pairs with a Python
`sum(x * y for x, y in zip(...))`. With `RETRIEVAL_CANDIDATES=50` and 384 dimensions that is
up to 1 225 pairs × 384 multiplications — roughly **470 000 Python-level operations per
context build**, on top of parsing 50 × 384 floats out of text. It was the largest
non-SQL cost in `build_context`.

Measured on 50 candidates × 384 dims, mean over 200 runs:

| Implementation | Time per call |
|---|---|
| Nested Python loops | 33.20 ms |
| Single matrix product (`M @ M.T`) | 0.94 ms |

**35× faster.** Same semantics, including the chain-breaking rule: only a *kept* memory can
cancel another one, so a cancelled memory never propagates cancellations. A randomised
equivalence test (`test_redondance_equivalente_a_la_reference_naive`) replays the naive
algorithm over 30 pseudo-random candidate sets and asserts identical scores, so the
vectorised version cannot drift.

## Related: sort on the operator, never on the alias

pgvector only uses an HNSW index for `ORDER BY embedding <=> $1` (ascending). `ORDER BY
similarity DESC`, where `similarity` is the alias of `1 - (embedding <=> $1)`, is **not**
recognised and falls back to a full scan. Three sites were affected and are now fixed:
`_fetch_candidates` (non-hybrid path), `retrieve_memories` (non-hybrid path), and
`_entangle` in the worker — the last one ran a full scan per extracted fact, so the cost of
building the entanglement graph grew linearly with the size of the memory.
