# Changelog

## 0.2.2 - Unreleased — performance, observabilité, outillage

Lot P1 de l'audit du 28/07. Aucune rupture de compatibilité d'API.

### Performance
- **La recherche hybride n'utilisait pas l'index HNSW.** La CTE `filtre` était référencée
  trois fois, donc matérialisée par PostgreSQL : le tri par distance portait sur un résultat
  intermédiaire et non sur la table. Mesuré sur 5 000 mémoires : `Seq Scan` + tri en
  **64–125 ms**, contre **6–9 ms** avec un `Index Scan using idx_memories_embedding_hnsw`.
  L'ancien plan était **linéaire en volume**. Preuve reproductible :
  `scripts/explain_retrieval.py`, plans commentés dans `benchmarks/EXPLAIN_retrieval.md`.
- **Tris vectoriels corrigés à trois endroits.** `ORDER BY similarity DESC` (alias) n'active
  pas l'index ; seul `ORDER BY embedding <=> $1` le fait. Corrigé dans `_fetch_candidates`,
  `retrieve_memories` et `_entangle` — ce dernier faisait un scan complet **par fait
  extrait**, donc un coût d'intrication croissant avec la taille de la mémoire.
- **Filtre de redondance vectorisé** : jusqu'à 1 225 paires × 384 multiplications en Python
  pur par requête (~470 k opérations, **33,2 ms**) remplacées par un produit matriciel
  numpy (**0,94 ms**, soit ×35). Sémantique strictement préservée, y compris la rupture de
  chaîne, verrouillée par un test d'équivalence randomisé contre l'implémentation naïve.
- **Index manquants** (`20260729_perf_idx`) : `relationships(target_memory_id)` — la clause
  `OR target_memory_id = ANY(...)` de `build_context` faisait un parcours séquentiel complet
  de la table des relations à chaque appel.
- **Cache d'authentification** (`AUTH_CACHE_TTL`, 60 s) : chaque requête faisait un
  `SELECT` + `UPDATE` + `COMMIT` sur la **même ligne** d'`api_keys`. ⚠️ Compromis explicite :
  une clé révoquée reste acceptée au maximum ce délai ; `AUTH_CACHE_TTL=0` rétablit la
  révocation immédiate.
- `parse_embedding` : conversion numpy en une passe au lieu de ~19 000 appels à `float()`
  par construction de contexte.

### Architecture
- **`build_context` extrait en couche service** : 208 lignes dans le handler HTTP →
  57 lignes, l'orchestration des 4 phases Q-EM vivant désormais dans
  `synaptiq_core.context_builder` avec un `MemoryStore` injecté. Testable sans HTTP ni
  PostgreSQL (13 nouveaux tests).
  Effet de bord voulu : un `MemoryStore` est construit pour UN couple (tenant, agent) et
  aucune de ses méthodes ne prend ces paramètres — **la fuite d'isolation F1 devient
  structurellement inexprimable**.

### Observabilité
- **Journalisation structurée JSON** (`synaptiq_core.observability`, sans dépendance) avec
  `LOG_FORMAT=json|text` et `LOG_LEVEL`.
- **Tracebacks conservés** : `exc_info=True` sur tous les gestionnaires qui journalisaient
  un échec. Une vingtaine faisaient `logger.error(f"...{e}")`, ce qui perd la pile entière.
- **`trace_id` réellement corrélable** : UUID par requête (il était dérivé d'un horodatage
  à la seconde, donc partagé par les requêtes concurrentes) et propagé par `contextvar`
  dans tous les logs, y compris ceux émis depuis `synaptiq_core`.
- **Jauges de santé du pipeline** sur `/metrics` : `synaptiq_outbox_pending`,
  `synaptiq_outbox_oldest_age_seconds` (révèle un relais mort — un compteur plat est
  indiscernable d'une absence de trafic) et `synaptiq_dlq_depth`.

### Mesure
- **`synaptiq_core.stats`** : intervalles de confiance de Wilson, verdict de significativité
  et dimensionnement d'échantillon. Le harness LOCOMO émet désormais l'incertitude EN MÊME
  TEMPS que l'exactitude : une proportion ne peut plus être publiée sans sa marge.
- **README corrigé** : le « +3,29 pts » annoncé portait sur 152 questions, où l'IC à 95 %
  vaut [−7,9 ; +14,5] — l'écart n'est pas significatif. Le tableau le dit maintenant
  explicitement, et indique qu'il faut ~2 400 questions pour une marge de ±2 points.
- `make bench` / `.\scripts\dev.ps1 bench` : entrée reproductible (graine fixe, tenant dédié).

### Outillage et conteneurs
- **Ruff élargi** de `E9,F` à `E,W,F,I,B,UP,S,C4,RUF` (265 corrections mécaniques
  appliquées, exemptions justifiées dans `ruff.toml`).
- **Mypy** sur `packages/core` (propre, 9 modules) — il a immédiatement révélé deux
  annotations **fausses** : `route_memory` déclarait `-> str` en renvoyant `None`, et
  `compute_recency_factor` refusait le `None` qu'il traite pourtant.
- **Couverture** de `packages/core` à **95 %**, seuil CI à 90 %.
- **Pre-commit** (`.pre-commit-config.yaml`) : ruff, mypy, hygiène de fichiers, détection de
  clés privées. Volontairement sans les tests, qu'un hook lent ferait contourner.
- **Tests MCP (10) et SDK Python (10)** : ces deux surfaces n'en avaient aucun.
- 🐛 **L'image Docker du worker était cassée.** `apps/worker/requirements.txt` avait dérivé
  du code : ni `prometheus-client` (importé depuis l'ajout du compteur d'extractions
  dégradées) ni `numpy`. Le conteneur échouait au démarrage sur
  `ModuleNotFoundError: No module named 'prometheus_client'`. Remplacé par
  `requirements-common.txt`, socle partagé API/worker/relais.
- **Conteneurs non root** (`USER synaptiq`) sur les trois images, et `.dockerignore`
  complété (tests, benchmarks, dataset et visuels n'ont plus à entrer dans les images).
- `Makefile` + `scripts/dev.ps1` (équivalent Windows, mêmes noms de cibles).

## 0.2.1 - Unreleased — durcissement sécurité & intégrité mémoire

Lot P0 de l'audit du 28/07. Deux ruptures de compatibilité, signalées ci-dessous.

### Sécurité
- **Isolation par agent appliquée dans la traversée du graphe.** `/context/build` complétait
  ses candidats par les mémoires « manquantes » du graphe d'intrication sans filtrer le
  tenant ni l'agent : une seule arête traversante injectait la mémoire d'un tiers dans le
  contexte envoyé au LLM.
- **Scopes de clé API** (`read` / `write` / `admin`) et **périmètre d'agents** (`agent_scope`).
  `agent_id` n'est plus une simple valeur de body : une clé bornée à un agent reçoit `403`
  si elle en cible un autre.
- **⚠️ Rupture — purge RGPD.** `DELETE /v1/memories` exige désormais le scope `admin` **et**
  `?confirm=<tenant_id>`, et écrit une ligne dans la nouvelle table `audit_log` (dans la
  même transaction). Les clés existantes conservent `read`+`write` mais perdent la purge :
  émettre une clé dédiée avec `create_api_key.py --scopes read write admin`.
- **⚠️ Rupture — outils MCP.** `agent_id` retiré des paramètres de `store_memory`,
  `recall_memories` et `build_context` : l'identité vient de `SYNAPTIQ_AGENT_ID` côté
  serveur. C'était le LLM qui choisissait son identité mémoire.

### Intégrité des données
- **Plus d'archivage sur la seule similarité.** Une nouvelle préférence archivait toute
  préférence active au-delà de 0,8 de cosinus, sans vérifier la moindre contradiction :
  « mails courts » et « mails en français » (~0,85) se supprimaient l'une l'autre en
  silence. L'archivage exige maintenant un verdict explicite
  (`synaptiq_core.contradiction`, juge LLM pluggable, **fail-closed** : juge en panne =
  rien d'archivé). Nouveau réglage `CONTRADICTION_JUDGE` (`auto` | `llm` | `off`) ;
  `CONTRADICTION_SIM_THRESHOLD` devient un simple pré-filtre. Chaque archivage tisse une
  arête `supersedes_by` traçable.

### Correctifs
- `/v1/events` renvoie `503` (et non plus `500`) quand PostgreSQL est indisponible : le
  `except Exception` ravalait le 503 et empêchait tout retry côté client.
- `get_auth` : une panne SQL laissait `row` non liée (`NameError` opaque).

### Schéma & outillage
- **Alembic est la seule autorité du schéma.** `infra/postgres/init.sql` réduit à
  `CREATE EXTENSION vector` (il dupliquait les migrations, et son `ADD CONSTRAINT` non
  idempotent faisait échouer tout rejeu) ; l'étape `psql init.sql` retirée de la CI.
- Nouvelle révision `20260729_key_scopes` : `api_keys.scopes`, `api_keys.agent_scope`,
  table `audit_log`.
- Tests : 130 unitaires + 23 d'intégration (153 au total), dont un test de non-régression
  par correctif de ce lot.

## 0.2.0 - Unreleased
- Transactional outbox et déduplication worker par événement source.
- Migrations Alembic, contrats API bornés et trace de retrieval optionnelle.
- Compose local sécurisé et SDK TypeScript initial.
