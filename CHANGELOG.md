# Changelog

## 0.2.6 - Unreleased — le lanceur attend son infrastructure

`start_services.ps1 -WaitForInfra <secondes>` patiente que PostgreSQL (5435) et Redis (6399)
repondent avant de demarrer l'API. Indispensable a l'ouverture de session : Docker Desktop met
souvent une a deux minutes a lever ses conteneurs, et une API demarree avant eux garde un pool
NULL et repond 503 sur tout — **sans jamais se retablir d'elle-meme**. Sans cette attente, un
demarrage automatique produisait donc une instance en apparence lancee mais totalement muette.

Sans le drapeau (lancement manuel), le comportement est inchange si l'infra est la, et un
refus explicite sinon, avec la commande a lancer — plutot qu'un demarrage voue a l'echec.

## 0.2.5 - Unreleased — l'API plantait au demarrage sur un .env non-ASCII (Windows)

Trouve en montant une instance de production sur Windows : `cp .env.example .env` puis
lancement local de l'API echouait sur un `UnicodeDecodeError` opaque, tres loin de sa cause.

`slowapi` relit `.env` de son cote, via `starlette.config.Config`, **sans specifier
d'encodage** : sur Windows c'est donc cp1252, et un seul octet UTF-8 non representable fait
tomber l'import. Un emoji suffit — l'octet `0x8f` du selecteur de variante `U+FE0F` — et le
`.env.example` livre en contient. Les accents seuls passent (mal decodes mais sans erreur),
ce qui rend le probleme d'autant plus deroutant.

Correctif a la racine : `Limiter(..., config_filename="")` empeche cette relecture, qui etait
de toute facon redondante — SynaptiQ charge sa configuration via `load_dotenv` (UTF-8) et
passe `default_limits` explicitement. Verifie avec un `.env` contenant un emoji : l'import
passe, sans avertissement.

## 0.2.4 - Unreleased — transport MCP : stdio vs HTTP

### Limite mesurée du transport stdio avec antigravity CLI

Après fermeture de stdin, `mcp.run()` met **141 à 250 ms** (médiane 157) à se dénouer. Le
temps est passé dans la boucle anyio de fastmcp, **pas** dans l'arrêt de l'interpréteur : un
`os._exit()` en sortie de `mcp.run()` n'y change rien (vérifié, puis retiré).

Or antigravity CLI n'accorde qu'une fenêtre de grâce d'environ 100 ms avant d'appeler
`Kill()`. Sur Windows, `TerminateProcess(handle, 1)` se lit `exit status 1`, et son
gestionnaire abandonne alors le rechargement de **tous** ses serveurs MCP. Les serveurs Node
passent sous cette limite, Python non.

Conséquence documentée : avec ce client, exposer le serveur en **HTTP** et le déclarer par
`serverUrl` côté client. Il n'y a alors plus de processus enfant à arrêter, donc plus de
fenêtre de grâce à respecter. C'est exactement ce qui a réglé le même symptôme sur le
serveur Obsidian, dont le pont `npx mcp-remote` ne terminait jamais son handshake.

`scripts/start_services.ps1` démarre l'API et le serveur MCP HTTP de façon idempotente
(`-Status`, `-Stop`), en attendant l'écoute effective plutôt qu'en annonçant un démarrage
optimiste.

## 0.2.3 - Unreleased — suites d'un incident de production

Trois correctifs issus d'une panne constatée le 29/07 sur une instance réelle : le serveur
MCP répondait « aucun souvenir trouvé » alors que la base en contenait.

### `SYNAPTIQ_AGENT_ID` devient obligatoire (rupture)
Cause racine de la panne. La variable valait `qwen_code_agent` par défaut, alors que les
souvenirs de l'instance avaient été écrits sous un autre identifiant. Le serveur lisait donc
une partition vide **sans lever d'erreur** — symptôme indiscernable d'une mémoire réellement
vide, et donc indébuggable de l'extérieur.

Elle n'a plus de défaut : le serveur MCP refuse de démarrer sans elle, avec un message qui
indique quoi faire et comment retrouver l'identifiant des souvenirs existants. Les outils
refusent aussi de partir sans identité plutôt que d'interroger une partition arbitraire.

### Plus aucun effet de bord à l'import du serveur MCP
`ensure_api_running()` était appelée au niveau module : importer `apps.mcp.server` démarrait
un uvicorn — y compris depuis la suite de tests. Déplacée dans `__main__`, désactivable par
`SYNAPTIQ_AUTOSTART_API=false`, et le sous-processus est maintenant détaché avec
stdout/stderr redirigés vers `api.log`. Ce dernier point n'est pas cosmétique : en transport
`stdio`, stdout porte le JSON-RPC du protocole MCP, et un uvicorn qui y écrit corrompt la
session.

Dans la même veine, `configure_logging()` accepte un `stream` : le serveur MCP journalise
sur **stderr**, jamais stdout.

**Correctif du correctif** : le déplacement dans `__main__` avait laissé l'attente de l'API
*bloquer le handshake MCP*. Mesuré : 14,06 s avant le handshake quand l'API était injoignable,
contre 1,9 s sinon. Le client MCP, dont le délai d'initialisation est bien plus court, tuait
alors le serveur — `exit status 1` sur Windows — et **le rechargement de tous ses serveurs
échouait**. Le démarrage de l'API ne bloque plus (`SYNAPTIQ_AUTOSTART_WAIT_S=0` par défaut) :
2,84 s de handshake API injoignable, code de sortie 0. Si le premier appel d'outil arrive
avant que l'API écoute, il réessaie une fois (`SYNAPTIQ_RETRY_DELAI_S`).

Leçon générale : **le handshake d'un protocole ne doit jamais attendre une tâche annexe.**

### `SYNAPTIQ_AGENT_ID` manquant ne tue plus le serveur

Rendre l'identité obligatoire était juste ; **échouer au démarrage** ne l'était pas. En
contexte MCP, un serveur qui refuse de démarrer disparaît de la liste du client, qui
n'affiche qu'un `failed to stop mcp instance: synaptiq: exit status 1` et jette stderr — le
message d'aide, soigneusement rédigé, n'atteignait donc personne. Symptôme constaté : le
serveur invisible et un code d'erreur opaque, soit l'inverse de l'intention.

Échouer vite n'a de valeur que si quelqu'un LIT l'échec. Le serveur démarre désormais
toujours et expose ses outils ; `verifier_configuration()` journalise un `DEMARRAGE
DEGRADE`, et chaque appel d'outil renvoie l'explication complète — le seul canal que
l'utilisateur lit réellement. Le refus de démarrer reste en place pour le transport
**réseau** sans clé API, où c'est une ouverture et non une gêne de diagnostic.

Seconde cause du même `exit status 1`, indépendante : `python -m apps.mcp.server` exige que
le client applique `cwd`, sans quoi c'est un `ModuleNotFoundError`. Le script insérant
lui-même `sys.path`, la configuration de référence l'appelle désormais par **chemin absolu**,
sans `cwd`.

### La taxonomie s'applique aux DEUX chemins d'écriture
`VALID_SUBTYPES` vivait dans le worker, donc n'était appliquée qu'à l'extraction LLM :
`POST /v1/memories` acceptait n'importe quel sous-type. Constaté en production, des mémoires
portaient `seo_audit_july_2026`, `nana_intelligence_lead_webhook`… Déplacée dans
`synaptiq_core.taxonomy`, partagée par les deux chemins.

La règle retenue n'est **pas** un rejet strict : un sous-type libre reste accepté (le routage
retombe proprement sur la collection du type, et des intégrations réelles en dépendent).
Seul un sous-type canonique rattaché au mauvais type est refusé en 422 — `type=semantic` avec
`subtype=coding_best_practices` irait dans `facts` alors que son auteur visait
`best_practices`, c'est la seule erreur que la validation puisse démontrer.

`POST /v1/memories` retourne désormais `collection` et `canonical_subtype`, pour que
l'appelant sache où son souvenir sera servi au lieu de le supposer.


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
