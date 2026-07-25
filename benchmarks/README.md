# Benchmarks SynaptiQ

Deux harnais, deux usages : un **micro-benchmark hors ligne** du collapse Q-EM, et un
**benchmark de bout en bout** sur LOCOMO qui mesure la chaîne complète (ingestion →
consolidation → rappel → réponse) contre une baseline vectorielle.

---

## 1. LOCOMO — mémoire long-terme conversationnelle

`locomo_runner.py` rejoue une conversation multi-sessions à travers le **vrai pipeline**
(worker d'extraction inclus), puis répond aux questions du dataset et fait juger les
réponses par un LLM. C'est le protocole dominant du domaine (« J-score »).

### Dataset

Le dataset n'est **pas versionné ici** : c'est une œuvre tierce distribuée sous sa propre
licence. Le télécharger depuis la source officielle :

```bash
curl -L -o benchmarks/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

10 conversations, ~5 900 tours, ~1 990 questions réparties en 5 catégories
(1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial — **exclue** par
convention, comme dans la littérature).

### Exécution

```bash
python benchmarks/locomo_runner.py benchmarks/locomo10.json \
    --conv 0 --arm both --top-k 50 --qa-workers 4 --out resultats.json
```

| Option | Rôle |
|---|---|
| `--arm qem\|vector\|both` | Moteur Q-EM, baseline top-k vectorielle, ou les deux |
| `--top-k` | Candidats de la baseline avant troncature au budget de tokens |
| `--qa-workers` | Questions évaluées en parallèle (l'ingestion reste séquentielle : l'ordre construit le graphe d'intrication) |
| `--resume` | Reprend une ingestion interrompue au lieu de repartir de zéro |
| `--max-degraded` | Part maximale d'extractions repliées sur les regex avant abandon (défaut 5 %) |
| `--limit-turns`, `--limit-qa` | Bornes pour un smoke test |

### Ce que le harness garantit

- **Comparaison à budget égal.** Les deux bras sont tronqués au même budget de tokens avec
  le **même estimateur** (`estimate_tokens`, celui du collapse). Sans cela, l'écart
  mesurerait la taille du contexte, pas la qualité du rappel.
- **Juge distinct du répondeur.** `LOCOMO_MODEL_QA` et `LOCOMO_MODEL_JUDGE` sont séparés,
  pour écarter l'auto-préférence. Le **répondeur doit être identique sur les deux bras**,
  sinon on mesure le modèle et non le moteur de mémoire.
- **Détection des corpus dégradés.** Le worker se replie silencieusement sur des
  heuristiques regex quand le LLM échoue (429, timeout…), ce qui produit des mémoires
  `episodic` exclues du graphe. Le harness compte ces replis, les publie
  (`degraded_ratio`) et **abandonne le run** au-delà du seuil : un score obtenu sur un
  corpus à moitié dégradé n'est pas publiable.
- **Reproductibilité.** Le rapport embarque les modèles, le modèle d'embedding et tous les
  seuils Q-EM effectifs — un score n'a de sens qu'accompagné de sa configuration.

### Fournisseur LLM

N'importe quel endpoint OpenAI-compatible via `LLM_BASE_URL`. Pour un run complet
(~1 000 appels), les tiers gratuits saturent vite : Groq plafonne à 8 000 tokens/minute,
ce qui déclenche des replis regex. `agy_openai_shim.py` (ci-dessous) contourne le problème.

---

## 2. Passerelle Antigravity — `agy_openai_shim.py`

Expose un endpoint **OpenAI-compatible** local qui relaie vers Antigravity CLI (`agy`),
lequel s'authentifie avec le compte Google de la machine : pas de clé API, pas de quota
par minute.

```bash
python benchmarks/agy_openai_shim.py --port 8899 --model gpt-oss-120b-medium
# puis dans .env :  LLM_BASE_URL=http://127.0.0.1:8899/v1
```

SynaptiQ ne connaît que du HTTP OpenAI-compatible : **aucun code produit n'est spécifique
à ce fournisseur**. La passerelle isole aussi un `HOME` dédié à `agy`, sans quoi le
`~/.gemini/GEMINI.md` de l'utilisateur (instructions personnelles) contaminerait chaque
extraction et les serveurs MCP démarreraient à chaque appel.

---

## 3. Micro-benchmark du collapse — `qem_vs_vector.py`

Compare, hors ligne, le collapse Q-EM à un top-k de même taille sur des candidats
pré-scorés. Aucune infra requise.

```bash
PYTHONPATH=packages/core python benchmarks/qem_vs_vector.py dataset.jsonl
```

Chaque ligne JSONL : `{"expected_ids": [...], "candidates": [{"id", "score", "content",
"type", "subtype"}]}`.

---

## Publier un résultat

Un score n'est exploitable qu'avec **le dataset, le modèle d'embedding, les modèles LLM et
les seuils Q-EM** qui l'ont produit — tous présents dans le rapport JSON. Y joindre le taux
d'extractions dégradées et le coût moyen en tokens de contexte : à exactitude comparable,
le contexte le plus court gagne.
