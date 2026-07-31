# Changelog

> **Versions anterieures a 0.3.0 : datees et taggees a posteriori.** Jusqu'au 31/07 rien
> n'avait ete tagge depuis `0.2.0`, et huit entrees portaient la mention « Unreleased ». La
> date de chaque version est celle du commit qui a livre son contenu, indique a cote du
> numero. Deux consequences a connaitre :
>
> - **`0.2.1` et `0.2.2` pointent sur le MEME commit** (`c5d2516`) : les lots P0 et P1 de
>   l'audit du 28/07 ont ete livres ensemble. Les separer ici garde la lisibilite du
>   decoupage, mais il n'y a jamais eu deux livraisons.
> - **Les tags ne partitionnent pas tout l'historique.** Quelques commits ne sont rattaches a
>   aucune entree — notamment le lot du 26/07 (recherche hybride, harnais LOCOMO, embeddings
>   OpenRouter) et `a0b844b`. Le journal avait ces trous avant les tags ; les combler
>   a posteriori aurait demande d'inventer des notes de version.

## 0.3.1 - Unreleased — les conteneurs se pointaient sur eux-memes

### Regression de `a0b844b` : `EMBEDDING_BASE_URL` en conteneur

`a0b844b` a rendu l'endpoint d'embedding surchargeable depuis `.env` en reutilisant le MEME
nom de variable : `${EMBEDDING_BASE_URL:-http://host.docker.internal:1234/v1}`. Or le `.env`
declare legitimement `EMBEDDING_BASE_URL=http://localhost:1234/v1` — valeur juste pour un
process de l'hote. Compose reprenait donc cette valeur, et dans le conteneur `localhost`
designe le conteneur lui-meme.

Consequence mesuree le 30/07 sur l'instance de production : `Connection refused` sur
`/v1/embeddings` a chaque evenement, cinq tentatives, puis DLQ. **Le chemin `/events` etait
entierement hors service en conteneur**, avec une configuration qui paraissait correcte.
`LLM_BASE_URL` echappait deja au piege grace a une variable distincte (`LLM_BASE_URL_DOCKER`) ;
c'est ce motif qui est generalise.

### Deuxieme occurrence, trouvee en verifiant la premiere

Le service `api` n'avait aucune surcharge de `LLM_BASE_URL`. Ce n'est pas anodin : l'API
appelle un LLM elle aussi — `POST /v1/memories` soumet les preferences proches au juge de
contradiction (`synaptiq_core.contradiction`). En conteneur, elle aurait interroge son propre
port. Le juge etant fail-closed, plus rien n'aurait jamais ete archive — sans erreur, encore
une fois.

`EMBEDDING_BASE_URL_DOCKER` et `LLM_BASE_URL_DOCKER` sont desormais posees sur `api` et
`worker`. Verifie par `docker compose config` sur trois scenarios : `.env` en `localhost` sans
surcharge (le cas de la regression), surcharge explicite vers un fournisseur distant
(l'objectif initial de `a0b844b`, preserve), et `.env` vide.

> **Obstacle independant, cote hote.** Si le serveur de modeles n'ecoute que sur `127.0.0.1`,
> aucun conteneur ne l'atteindra meme avec la bonne URL. Dans LM Studio, activer « Serve on
> Local Network ».

### MCP : une panne ne doit plus ressembler a un souvenir

Remonte de l'instance de production, ou le correctif avait ete ecrit en local.

Les outils MCP renvoyaient `"[ERROR] ..."` pour **tout** echec. Cote client, cette chaine est
un resultat valide : le 30/07, un `Read timed out` s'est affiche dans la conversation de
l'agent comme s'il s'agissait du contenu de la memoire. Une panne sort desormais en
`ToolError` (`isError: true`) sur les six outils.

L'exception est deliberee : une **identite manquante** (`SYNAPTIQ_AGENT_ID` absent) reste un
message texte, parce que c'est un defaut de configuration dont le message porte la marche a
suivre — et non une panne.

Cause de ce timeout : `_poster`/`_lire` plafonnaient a **5 s en dur**, alors que le premier
appel d'une session paie le chargement du modele d'embedding (mesure a plus de 5 s a froid,
~2,6 s ensuite). Le premier `recall_memories` de chaque session echouait donc — precisement
celui qui construit le contexte initial de l'agent. Nouveau defaut : `SYNAPTIQ_TIMEOUT_S=30`.

## 0.3.0 — 2026-07-31 — l'agent structure sa propre memoire

Jusqu'ici une « collection » n'existait nulle part : c'etait le resultat d'une cascade de
`if` dans `route_memory`. Un agent ne pouvait ni la consulter, ni en creer, ni decider
comment ses souvenirs sont ranges — alors que SynaptiQ est cense etre SA memoire. Il
inventait bien des sous-types (`nana_intelligence_lead_webhook` existe en production), mais
ils etaient servis dans `facts` comme n'importe quel fait, et **rien ne le lui disait**.

### Le partage des roles : famille au moteur, collection a l'agent

C'est l'invariant de tout le lot, et il tient en une phrase : **le `type` n'est pas une
etiquette, c'est un comportement.**

- `memories.type` -> la **FAMILLE** (`semantic`, `episodic`, `procedural`, `working`).
  Fermee, propriete du moteur : elle decide de l'intrication, de la decroissance et de la
  section de repli.
- `memories.subtype` -> le **NOM de la collection**. Libre, propriete de l'agent.

Ce decoupage epouse le schema existant, donc **aucune donnee n'est deplacee** : les
sous-types deja ecrits deviennent retroactivement de vraies collections. La migration
`20260731_collections` les declare, verifiee sur base vierge ET sur base peuplee.

### Ce que l'agent peut faire maintenant

Trois outils MCP (`list_collections`, `create_collection`, `merge_collections`) et quatre
routes (`GET`/`POST /v1/collections`, `POST /v1/collections/merge`, plus le filtre
`collections` sur `/v1/retrieve` et `/v1/context/build`).

La creation est **explicite** : ecrire dans une collection inexistante ne la cree pas. Un
rangement auto-cree serait indiscernable d'une faute de frappe — `clients_paca`,
`client_paca` et `clientspaca` cohabiteraient, chacun avec sa section, sans qu'on sache
laquelle fait foi.

### RUPTURE — `context_packet` n'a plus un nombre de cles fixe

Les sept sections canoniques restent toujours presentes, **meme vides**, et chaque
collection declaree en ajoute une. Un consommateur qui lit sept cles en dur doit passer a
une iteration sur les entrees. Les deux SDK sont mis a jour et documentent la rupture ; les
deux harnais de benchmark iteraient deja dynamiquement.

La forme ne depend PAS de la presence de resultats : une recherche infructueuse renvoie les
memes cles qu'une recherche fructueuse. Sans cela, le consommateur aurait du tester
l'existence de chaque cle.

### Le gain qui touche le rappel : l'intrication par collection

`QEM_ENTANGLE_TYPES` etait un reglage d'INSTANCE : `episodic` ne tissait **aucune arete,
pour personne**. Bon defaut — les episodes bruts sont nombreux et peu discriminants — mais
impossible a nuancer. Un agent peut desormais declarer qu'une collection d'episodes est
structurante (des comptes rendus de reunion) et la faire alimenter le graphe, sans que les
journaux bruts le polluent. Le multi-hop est la seule dimension ou Q-EM creuse nettement
l'ecart sur la baseline : c'est le seul levier de ce lot qui touche la qualite du rappel.

### Deux silences fermes

- **`route_memory` ne renvoie plus jamais `None`.** Il le faisait pour une famille inconnue,
  et `collapse_by_utility` retirait alors la memoire du paquet — *apres* l'avoir comptee
  dans `selected_ids` et avoir depense son budget de tokens. Retrouvee, payee, invisible.
- **`store_memory` (MCP) rend son verdict de rangement.** L'API renvoyait `collection` et
  `canonical_subtype` depuis toujours ; l'outil les jetait et repondait exactement la meme
  chose qu'un rangement reussi. L'agent n'avait aucun moyen de distinguer les deux, donc
  aucun moyen d'apprendre.

Corrige au passage : l'outil MCP `build_context` iterait sur sept libelles codes en dur, ce
qui aurait ecarte **en silence** toutes les sections creees par l'agent — son propre
rangement invisible dans son propre contexte.

### Garder une taxonomie auto-construite lisible

Un LLM a qui l'on donne le droit de creer une categorie en cree une par nuance. Quatre
garde-fous, aucun optionnel :

- **anti-doublon SEMANTIQUE** (`COLLECTION_DUP_THRESHOLD`, 0,85) : les descriptions sont
  vectorisees et comparees, et le refus **nomme** la collection proche. L'unicite du nom ne
  protege de rien ici. Les descriptions systeme, sans vecteur en base, sont embarquees a la
  volee — sinon la protection serait inoperante contre le cas le plus probable, un agent
  debutant qui redouble un des sept rayons livres ;
- **plafond** (`MAX_COLLECTIONS_PER_AGENT`, 50), expose dans la liste pour etre anticipe ;
- **fusion** : sans elle une taxonomie ne peut que grossir. Les souvenirs changent
  d'etiquette, jamais detruits. Collections systeme et fusions inter-familles refusees — la
  famille porte un comportement, la changer ne serait pas qu'un rangement ;
- **dormantes** : vide au-dela de `COLLECTION_STALE_DAYS` (14) -> signalee, avec l'issue
  nommee. Un defaut qu'on ne voit pas ne se corrige pas.

**Pas de plafond sur le nombre de sections du paquet**, contrairement a ce qui etait prevu :
verification faite, le rendu n'imprime que les sections ayant du contenu, donc 40
collections ne produisent aucune rubrique vide dans le prompt — et un plafond aurait casse
la garantie de forme stable. Un test verrouille cette propriete.

Suite : **377 tests** (307 unitaires + 70 integration), ruff et mypy propres, couverture de
`packages/core` a 96 %. Verifie aussi sous `SYNAPTIQ_AUTH_REQUIRED=true` et
`RETRIEVAL_HYBRID=false` — le filtre de collection est repete dans les deux branches SQL,
comme le filtre tenant/agent.

## 0.2.7 — 2026-07-30 · `2e02463` — audit d'exploitation : quatre pannes silencieuses

Audit du 30/07 mene sur le depot ET sur une instance de production reelle. Le code s'en sort
bien ; l'exploitation beaucoup moins. Les quatre defauts trouves ont le meme trait : **aucun
d'eux ne produit d'erreur**. C'est ce qui les rend graves pour un moteur de memoire, ou un
resultat vide est indiscernable d'une memoire vide.

### RUPTURE DE COMPATIBILITE — `admin` n'est plus implicite sans authentification

`SYNAPTIQ_AUTH_REQUIRED=false` ouvrait la **purge RGPD**. Constate sur l'instance :
`DELETE /v1/memories?confirm=<mauvais>` repondait `400`, pas `401`, sans aucune cle. Chaine
complete : `get_auth()` rend `None`, `require_scope()` retournait aussitot, et il ne restait
qu'a connaitre le nom du tenant — que le message d'erreur `400` livrait lui-meme
(« Rappeler avec ?confirm=default »). Un appel local suffisait a vider l'instance.

`require_scope(None, "admin")` leve desormais `403`. `read` et `write` restent permis sans cle :
la commodite du mode de confiance vaut pour la lecture et l'ecriture, jamais pour la
destruction. **Impact** : un script qui purgeait sans cle doit maintenant en presenter une
portant `admin` (`create_api_key.py --scopes read write admin`).

### Les images Docker `api`, `relay` et `migrate` etaient inconstructibles

`apps/api/Dockerfile` copiait `requirements.txt` seul, alors que celui-ci commence par
`-r requirements-common.txt` depuis le socle partage (F19). Tout build echouait sur
`Could not open requirements file: '/app/requirements-common.txt'`. Comme cette image sert
aussi `relay` et `migrate`, les trois services etaient hors service — et personne ne l'a vu
parce que les conteneurs de l'instance tournaient encore sur des images anterieures au
refactor. Corollaire mesure : la base etait estampillee `20260729_perf_idx` tandis que l'image
`migrate` ignorait cette revision, donc `docker compose up` ne relevait plus rien
(`depends_on: migrate: service_completed_successfully`).

`apps/api/requirements.txt` est supprime : plus reference par rien, il avait deja derive
(ni `numpy`, ni `prometheus-client`, ni `alembic`, et `python-dotenv==1.0.1` — l'epinglage que
`requirements-common.txt` documente comme non resoluble). C'est exactement la derive que F19
avait corrigee sur le worker, laissee en place sur l'API.

### `/v1/health` denonce une ingestion a l'arret

Un relais mort rend le chemin `/events` **totalement silencieux** : l'API ecrit dans l'outbox,
repond `201 captured`, et la donnee n'est jamais consolidee. Constate sur l'instance, ou
`synaptiq-relay` etait eteint depuis deux jours. Les jauges qui le detectaient
(`synaptiq_outbox_pending`, `synaptiq_outbox_oldest_age_seconds`) existaient depuis F14, mais
rien ne scrutait `/metrics` : une supervision non branchee est une supervision absente.

`/v1/health` porte donc un troisieme service, `ingestion` : `healthy`, `stalled` ou `unknown`.
C'est l'AGE du plus vieil evenement non publie qui decide (`HEALTH_OUTBOX_MAX_AGE_S`, 300 s),
et non leur nombre : un outbox charge qui se vide est sain, un outbox a une entree immobile
depuis une heure ne l'est pas. Le `status` global bascule en `degraded` — donc le healthcheck
Compose de l'API le voit aussi.

### Le worker refuse de demarrer sur un modele d'embedding incompatible

L'instance a tourne avec `all-minilm-l6-v2` (anglophone) sur une base ecrite par
`paraphrase-multilingual-minilm-l12-v2`. **384 dimensions tous les deux** : aucune exception,
aucun log, aucune metrique — les vecteurs cessent simplement d'etre comparables et le rappel se
degrade en silence. `EMBEDDING_DIM` ne protege que du cas bruyant.

Le worker compare desormais, au demarrage, un vecteur deja stocke au meme contenu re-embarque
par le modele courant. Cosinus < `EMBEDDING_COHERENCE_MIN` (0,999) -> `SystemExit` avec la
mesure et la marche a suivre. Base vierge, base injoignable ou embedder muet : on laisse
passer, il n'y a rien a contredire. Echappatoire assumee : `EMBEDDING_COHERENCE_CHECK=false`.
Le worker QUITTE la ou le serveur MCP demarre degrade — la difference est qu'un worker qui
continue **ecrit** des vecteurs incompatibles, donc aggrave les degats a chaque evenement.

### Le relais ne s'empoisonne plus apres une coupure PostgreSQL

`relay.py` appelait `conn.rollback()` sans garde. Quand PostgreSQL avait coupe, ce rollback
levait `InterfaceError: connection already closed` — qui **remplacait l'erreur d'origine** dans
les journaux — puis `putconn()` rendait au pool une connexion MORTE, redistribuee ensuite
indefiniment : le relais ne se retablissait plus, meme PostgreSQL revenu. C'est le traceback
reel du conteneur mort le 28/07. `worker.py` traitait deja ce cas correctement ; les deux
formes sont alignees.

### Documentation

- Nouvelle section README **« Installing the MCP Server »** : ordre d'installation, tableau des
  variables, comparatif `stdio` / `http` avec la limite mesuree, installation de reference
  complete sous Windows, verification en trois points et tableau de depannage.
- Le README affirmait que le serveur MCP **refuse de demarrer** sans `SYNAPTIQ_AGENT_ID`. Le
  code fait deliberement l'inverse (et explique pourquoi) : corrige.
- `examples/claude_desktop_config.json` faisait exactement ce que le README interdit
  (`-m apps.mcp.server` + `cwd`), avec un chemin absolu personnel et l'`agent_id` a l'origine
  de la panne du 29/07 : reecrit.

Suite : **272 tests** (240 unitaires + 32 integration), ruff et mypy propres, couverture de
`packages/core` a 95 %.

## 0.2.6 — 2026-07-29 · `e6e0baf` — le lanceur attend son infrastructure

`start_services.ps1 -WaitForInfra <secondes>` patiente que PostgreSQL (5435) et Redis (6399)
repondent avant de demarrer l'API. Indispensable a l'ouverture de session : Docker Desktop met
souvent une a deux minutes a lever ses conteneurs, et une API demarree avant eux garde un pool
NULL et repond 503 sur tout — **sans jamais se retablir d'elle-meme**. Sans cette attente, un
demarrage automatique produisait donc une instance en apparence lancee mais totalement muette.

Sans le drapeau (lancement manuel), le comportement est inchange si l'infra est la, et un
refus explicite sinon, avec la commande a lancer — plutot qu'un demarrage voue a l'echec.

## 0.2.5 — 2026-07-29 · `d4c10c9` — l'API plantait au demarrage sur un .env non-ASCII (Windows)

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

## 0.2.4 — 2026-07-29 · `0c84dfd` — transport MCP : stdio vs HTTP

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

## 0.2.3 — 2026-07-29 · `5ddd040` — suites d'un incident de production

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


## 0.2.2 — 2026-07-29 · `c5d2516` — performance, observabilité, outillage

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

## 0.2.1 — 2026-07-29 · `c5d2516` — durcissement sécurité & intégrité mémoire

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

## 0.2.0 — 2026-07-24 · `681ef85`
- Transactional outbox et déduplication worker par événement source.
- Migrations Alembic, contrats API bornés et trace de retrieval optionnelle.
- Compose local sécurisé et SDK TypeScript initial.
