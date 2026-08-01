"""Bras `mem0` du harness LOCOMO — la référence open source, à conditions égales.

## Pourquoi ce bras existe

Le harness comparait Q-EM à une baseline top-k vectorielle. C'est le bon témoin pour isoler
l'apport du moteur, mais ce n'est pas la question que pose un lecteur : il veut savoir ce
que vaut SynaptiQ face à **mem0**, la référence de fait du créneau (62 k étoiles).

Les chiffres publiés de part et d'autre ne sont PAS comparables : juge différent, modèle
répondeur différent, sous-ensemble de questions différent. Le seul chiffre défendable est
un run des deux moteurs **dans le même harness**, ce que fait ce module.

## Ce qui est tenu identique — c'est tout l'intérêt

| Élément | Comment l'égalité est obtenue |
|---|---|
| Modèle d'embedding | `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` du `.env`, injectés dans mem0 |
| LLM d'extraction | `LLM_MODEL` / `LLM_BASE_URL`, injectés dans mem0 |
| Modèle répondeur | `LOCOMO_MODEL_QA` — appelé par le harness, en dehors des deux moteurs |
| Juge | `LOCOMO_MODEL_JUDGE` — idem |
| Stockage | pgvector, même serveur PostgreSQL, index HNSW |
| Contenu ingéré | la MÊME chaîne `[date] locuteur: texte`, dans le même ordre |
| Budget de contexte | `fit_to_budget`, l'estimateur du collapse Q-EM |

## Les asymétries qui subsistent — à citer avec tout résultat

1. **Datation.** SynaptiQ reçoit la date de session en `created_at` (elle alimente la
   décroissance temporelle et la résolution des questions temporelles). mem0 ne la reçoit
   qu'inline dans le texte et en métadonnée. Avantage structurel à SynaptiQ sur la
   catégorie « temporal », à ne pas passer sous silence.
2. **Ce qui est mesuré côté mem0 est le SDK OPEN SOURCE.** Les scores LoCoMo que mem0
   publie proviennent de leur plateforme managée, qui embarque, selon leur propre note,
   des « optimisations propriétaires absentes du SDK open source ». Comparer à la
   plateforme n'aurait aucun sens ici : elle n'est pas auto-hébergeable, donc hors du
   périmètre produit de SynaptiQ.
3. **Latence non comparable.** mem0 v3 fait son appel d'extraction *dans* `add()` ;
   SynaptiQ consolide en asynchrone derrière l'outbox. Le coût en appels LLM est
   comparable, le temps de réponse perçu ne l'est pas.
4. **Sans spaCy ET `en_core_web_sm`, mem0 est handicapé** : v3 fusionne sémantique + BM25 +
   entités, et les deux derniers signaux passent par ce modèle. Il manque, mem0 continue
   en silence, purement sémantique. `stats()["nlp"]` remonte l'état réel pour qu'un run
   dégradé ne soit pas publié comme un run propre.
5. **`en_core_web_sm` est un modèle ANGLOPHONE.** LOCOMO étant en anglais, c'est le bon
   choix ici — mais cela signifie que ce harness ne dit rien du comportement de mem0 sur
   un corpus français, là où SynaptiQ tourne avec un embedder multilingue.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import logging
import os
import sys
import threading
from urllib.parse import unquote, urlparse

import psycopg2
from psycopg2 import sql

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "packages", "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.budget import fit_to_budget

log = logging.getLogger("locomo.mem0")

# Préfixe imposé des tables mem0 : `reset()` supprime TOUT ce qui le porte, la contrainte
# est donc ce qui empêche un nom mal choisi d'emporter une table du produit.
COLLECTION_PREFIX = "mem0"
DEFAULT_COLLECTION = "mem0_bench_locomo"


class Mem0Unavailable(RuntimeError):
    """mem0 n'est pas installable/importable — message actionnable plutôt qu'un ImportError nu."""


def _charger_memory_class():
    try:
        from mem0 import Memory
    except ImportError as exc:  # pragma: no cover — dépend de l'environnement
        raise Mem0Unavailable(
            "Le bras mem0 exige le SDK open source :\n"
            "    pip install 'mem0ai[nlp]'\n"
            "    python -m spacy download en_core_web_sm\n"
            "Les deux lignes comptent. L'extra [nlp] n'installe QUE spaCy ; sans le modèle "
            "en_core_web_sm, mem0 v3 perd sa lemmatisation BM25 et sa liaison d'entités "
            "SANS lever d'erreur. On mesurerait alors un mem0 amputé de deux de ses trois "
            "signaux de rappel — un bras de paille, pas une comparaison."
        ) from exc
    return Memory


def capacites_nlp() -> dict:
    """mem0 v3 tourne-t-il à pleine capacité ? (vérifié dans le SDK installé, pas déduit)

    Depuis la v3, mem0 fusionne trois signaux : sémantique, BM25 et liaison d'entités. Les
    deux derniers passent par spaCy **et** par le modèle `en_core_web_sm` :

    - `mem0.utils.lemmatization.lemmatize_for_bm25` rend son entrée TELLE QUELLE quand le
      modèle n'est pas chargeable (`get_nlp_lemma()` renvoie None) ;
    - l'extraction d'entités renvoie alors une liste vide.

    Dans les deux cas, aucune exception : mem0 continue, en purement sémantique. Un score
    obtenu ainsi n'est pas celui de mem0, c'est celui d'un mem0 amputé — d'où la présence
    de cette information dans le rapport plutôt que dans un commentaire.

    ⚠️ mem0 tente de télécharger `en_core_web_sm` au premier usage. Sur une machine sans
    accès réseau, la tentative échoue en `warning` et la dégradation passe inaperçue.
    """
    try:
        import spacy
    except ImportError:
        return {"spacy": False, "en_core_web_sm": False, "full_capacity": False}
    modele = bool(spacy.util.is_package("en_core_web_sm"))
    return {"spacy": True, "en_core_web_sm": modele, "full_capacity": modele}


def _version_mem0() -> str:
    try:
        return importlib.metadata.version("mem0ai")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "inconnue"


def _dsn_vers_pgvector(dsn: str, collection_name: str, embedding_dim: int) -> dict:
    """Traduit le DSN SynaptiQ en configuration pgvector mem0 (mêmes serveur et base).

    Les deux moteurs partagent volontairement le même PostgreSQL : un écart de matériel ou
    de configuration de base fausserait la comparaison de latence autant que celle de
    rappel. `hnsw=True` aligne mem0 sur l'index que SynaptiQ utilise (cf. `20260729_perf_idx`).
    """
    u = urlparse(dsn)
    if not u.hostname:
        raise ValueError(f"DSN PostgreSQL inexploitable pour le bras mem0 : {dsn!r}")
    return {
        "dbname": (u.path or "/postgres").lstrip("/"),
        "user": unquote(u.username or ""),
        "password": unquote(u.password or ""),
        "host": u.hostname,
        "port": u.port or 5432,
        "collection_name": collection_name,
        "embedding_model_dims": embedding_dim,
        "hnsw": True,
    }


def reset_collection(dsn: str, collection_name: str) -> list[str]:
    """Supprime les tables du périmètre mem0. À appeler AVANT d'instancier `Memory`.

    Ces tables ne sont pas décrites par les migrations Alembic : le SDK mem0 les crée
    lui-même au premier usage. Elles vivent dans la même base pour que les deux moteurs
    partagent le même matériel, mais elles restent étrangères au schéma du produit — d'où
    ce nettoyage explicite, borné au préfixe dédié.

    Le balayage se fait par PRÉFIXE et non par nom exact : mem0 v3 crée une collection
    parallèle pour les entités. En ne supprimant que la table principale, le run suivant
    repartirait de mémoires vides tout en héritant des entités du run précédent — un état
    incohérent qui ne lèverait aucune erreur.

    Retourne les noms supprimés (le runner les journalise).
    """
    if not collection_name.startswith(COLLECTION_PREFIX):
        raise ValueError(
            f"Le nom de collection mem0 doit commencer par {COLLECTION_PREFIX!r} : cette "
            "fonction supprime TOUTES les tables publiques qui portent le préfixe donné, "
            "et un préfixe trop large emporterait des tables du produit."
        )
    db = psycopg2.connect(dsn)
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename LIKE %s",
                (f"{collection_name}%",),
            )
            tables = [r[0] for r in cur.fetchall()]
            for table in tables:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(table)))
            db.commit()
        if tables:
            log.info("Périmètre mem0 réinitialisé : %s", ", ".join(tables))
        return tables
    finally:
        db.close()


def build_config(*, dsn: str, collection_name: str, embedding_provider: str,
                 embedding_model: str, embedding_base_url: str, embedding_api_key: str,
                 embedding_dim: int, llm_model: str, llm_base_url: str,
                 llm_api_key: str) -> dict:
    """Configuration mem0 dérivée des variables d'environnement de SynaptiQ.

    Rien n'est codé en dur ici : c'est ce qui garantit que les deux moteurs voient le même
    modèle d'embedding et le même LLM d'extraction. Changer le `.env` déplace les deux bras
    ensemble, jamais un seul.
    """
    if embedding_provider == "lmstudio":
        # mem0 a un fournisseur LM Studio dédié ; le client OpenAI générique envoie un
        # paramètre `dimensions` que LM Studio rejette selon les modèles.
        embedder = {"provider": "lmstudio",
                    "config": {"model": embedding_model,
                               "lmstudio_base_url": embedding_base_url,
                               "embedding_dims": embedding_dim}}
    else:
        embedder = {"provider": "openai",
                    "config": {"model": embedding_model,
                               "openai_base_url": embedding_base_url,
                               "api_key": embedding_api_key or "local",
                               "embedding_dims": embedding_dim}}
    return {
        "vector_store": {"provider": "pgvector",
                         "config": _dsn_vers_pgvector(dsn, collection_name, embedding_dim)},
        "llm": {"provider": "openai",
                "config": {"model": llm_model,
                           "openai_base_url": llm_base_url,
                           "api_key": llm_api_key or "local",
                           # Température 0 des deux côtés : un benchmark reproductible ne
                           # peut pas laisser l'extraction varier d'un run à l'autre.
                           "temperature": 0.0}},
        "embedder": embedder,
    }


class Mem0Arm:
    """Adaptateur mem0 pour le harness : ingestion, rappel, comptages, remise à zéro."""

    def __init__(self, *, config: dict, user_id: str, dsn: str,
                 collection_name: str = DEFAULT_COLLECTION):
        if not collection_name.startswith(COLLECTION_PREFIX):
            raise ValueError(
                f"Le nom de collection mem0 doit commencer par {COLLECTION_PREFIX!r} : "
                "`reset()` supprime toutes les tables qui portent ce préfixe."
            )
        self.user_id = user_id
        self.dsn = dsn
        self.collection_name = collection_name
        self.config = config
        self.add_failures = 0
        self.search_failures = 0

        self.env_neutralisees = self._neutraliser_env_qui_detourne_le_llm(config)
        # mem0 lit parfois OPENAI_API_KEY directement dans l'environnement ; un endpoint
        # local (LM Studio, passerelle antigravity) n'en exige aucune, mais le client
        # refuse de s'instancier sans valeur.
        os.environ.setdefault("OPENAI_API_KEY", config["llm"]["config"].get("api_key") or "local")
        # Pas de télémétrie pendant une mesure : un appel réseau sortant ajoute de la
        # variance de latence et envoie des métadonnées de run à un tiers.
        os.environ.setdefault("MEM0_TELEMETRY", "false")

        Memory = _charger_memory_class()
        self._memory = Memory.from_config(config)

        # Les signatures de mem0 bougent à chaque version mineure (v3 a déplacé l'identité
        # de `search` dans `filters`). On inspecte une fois au démarrage plutôt que de
        # rattraper des TypeError à chaque appel, ce qui masquerait de vraies erreurs.
        self._add_accepte_metadata = self._accepte(self._memory.add, "metadata")
        self._search_par_filters = self._accepte(self._memory.search, "filters")

        # mem0 n'annonce pas sa thread-safety, et le pool pgvector est borné. Le harness
        # évalue les questions en parallèle : on sérialise le seul appel au SDK. L'appel
        # LLM (le coût réel) reste hors du verrou, la parallélisation garde son intérêt.
        self._verrou = threading.Lock()

    @staticmethod
    def _neutraliser_env_qui_detourne_le_llm(config: dict) -> list[str]:
        """Empêche l'environnement de rediriger le LLM de mem0 ailleurs que sur `LLM_BASE_URL`.

        ⚠️ **Piège constaté en vrai, et il ruine la comparaison sans rien signaler.**
        `mem0/llms/openai.py` teste `OPENROUTER_API_KEY` AVANT de regarder la configuration :

            if os.environ.get("OPENROUTER_API_KEY"):   # → tout part chez OpenRouter
                ...
            else:
                base_url = self.config.openai_base_url or os.getenv("OPENAI_BASE_URL") or ...

        Une variable présente pour un tout autre projet suffit donc à envoyer l'extraction
        mem0 chez un fournisseur distant pendant que SynaptiQ extrait en local. Ici, le
        modèle local n'existait pas chez OpenRouter et l'erreur a été bruyante (400) —
        c'est une chance. Avec un nom de modèle qui existe des deux côtés, le run aurait
        abouti et le rapport aurait affiché « même LLM d'extraction » en toute bonne foi.

        La variable est retirée du processus du benchmark (jamais de la machine), et
        `OPENAI_BASE_URL` est fixée à la valeur voulue plutôt que laissée au hasard de
        l'environnement. Retourne la liste des variables neutralisées, pour le rapport.
        """
        neutralisees = []
        if os.environ.pop("OPENROUTER_API_KEY", None) is not None:
            neutralisees.append("OPENROUTER_API_KEY")
            log.warning(
                "OPENROUTER_API_KEY retirée de l'environnement du benchmark : mem0 lui donne "
                "priorité sur la configuration et aurait envoyé l'extraction chez OpenRouter "
                "au lieu de %s.", config["llm"]["config"].get("openai_base_url"))
        voulue = config["llm"]["config"].get("openai_base_url")
        if voulue and os.environ.get("OPENAI_BASE_URL") not in (None, voulue):
            neutralisees.append("OPENAI_BASE_URL")
            log.warning("OPENAI_BASE_URL forcée sur %s (valeur d'environnement écartée).", voulue)
        if voulue:
            os.environ["OPENAI_BASE_URL"] = voulue
        return neutralisees

    @staticmethod
    def _accepte(fonction, nom: str) -> bool:
        try:
            params = inspect.signature(fonction).parameters
        except (TypeError, ValueError):  # pragma: no cover — builtins/objets exotiques
            return False
        if nom in params:
            return True
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    # ─── Cycle de vie ───

    @classmethod
    def from_env(cls, *, user_id: str, dsn: str,
                 collection_name: str = DEFAULT_COLLECTION) -> Mem0Arm:
        """Instancie le bras à partir du `.env` racine, celui que lisent déjà les autres bras."""
        config = build_config(
            dsn=dsn,
            collection_name=collection_name,
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "lmstudio"),
            embedding_model=os.getenv("EMBEDDING_MODEL", ""),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
        )
        return cls(config=config, user_id=user_id, dsn=dsn, collection_name=collection_name)

    def reset(self) -> None:
        """Alias d'instance de `reset_collection` (voir cette fonction)."""
        reset_collection(self.dsn, self.collection_name)

    # ─── Ingestion ───

    def ingest_turn(self, content: str, *, session_id: str, date: str) -> None:
        """Ingère UN tour de dialogue — la même chaîne que celle donnée à SynaptiQ.

        Un échec est compté, pas propagé : une coupure de rate limit en milieu de run ne
        doit pas détruire l'ingestion déjà faite. Le compteur remonte dans le rapport, et
        le runner interrompt le run si la part d'échecs dépasse le seuil — un corpus
        troué produirait un score bas qui ne mesurerait rien.
        """
        messages = [{"role": "user", "content": content}]
        kwargs: dict = {"user_id": self.user_id}
        if self._add_accepte_metadata:
            kwargs["metadata"] = {"session": session_id, "date": date}
        try:
            with self._verrou:
                self._memory.add(messages, **kwargs)
        except Exception:
            self.add_failures += 1
            log.warning("mem0 : échec d'ajout (%d au total)", self.add_failures, exc_info=True)

    # ─── Rappel ───

    def context(self, question: str, max_tokens: int, top_k: int) -> tuple[str, int]:
        """BRAS mem0 — `search()` puis troncature au budget commun.

        `top_k` est le même que celui de la baseline vectorielle : les trois bras partent
        du même nombre de candidats avant que le budget ne tranche.
        """
        try:
            with self._verrou:
                if self._search_par_filters:
                    brut = self._memory.search(
                        question, filters={"user_id": self.user_id}, limit=top_k)
                else:
                    brut = self._memory.search(question, user_id=self.user_id, limit=top_k)
        except Exception:
            self.search_failures += 1
            log.warning("mem0 : échec de recherche (%d au total)", self.search_failures,
                        exc_info=True)
            return "", 0
        return fit_to_budget(extraire_contenus(brut), max_tokens)

    # ─── Observation ───

    def memories_stored(self) -> int:
        """Nombre de mémoires consolidées par mem0, lu directement en base."""
        db = psycopg2.connect(self.dsn)
        try:
            with db.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (f"public.{self.collection_name}",))
                if cur.fetchone()[0] is None:
                    return 0
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(
                    sql.Identifier(self.collection_name)))
                return cur.fetchone()[0]
        finally:
            db.close()

    def stats(self) -> dict:
        """Tout ce qu'un rapport doit contenir pour être reproductible et lisible."""
        return {
            "mem0_version": _version_mem0(),
            # Sans spaCy ET son modèle, mem0 v3 perd BM25 et la liaison d'entités : un score
            # obtenu ainsi n'est PAS le score de mem0, et le rapport doit le dire.
            "nlp": capacites_nlp(),
            "collection_name": self.collection_name,
            "embedder": self.config["embedder"]["provider"],
            "embedding_model": self.config["embedder"]["config"].get("model"),
            "llm_model": self.config["llm"]["config"].get("model"),
            # L'URL réellement utilisée, pas celle demandée : c'est la seule preuve que
            # l'extraction mem0 et l'extraction SynaptiQ sont parties au même endroit.
            "llm_base_url": self.config["llm"]["config"].get("openai_base_url"),
            "env_neutralized": self.env_neutralisees,
            "memories_stored": self.memories_stored(),
            "add_failures": self.add_failures,
            "search_failures": self.search_failures,
        }


def extraire_contenus(brut) -> list[str]:
    """Normalise la réponse de `search()` en liste de textes.

    mem0 a rendu tantôt une liste, tantôt `{"results": [...]}` selon la version et le
    réglage `version`. Une erreur de forme ici ne lèverait aucune exception — elle rendrait
    simplement un contexte vide, et mem0 obtiendrait 0 % sans que rien ne le signale.
    C'est le mode de défaillance le plus coûteux d'un benchmark, d'où cette fonction isolée
    et testée séparément.
    """
    if isinstance(brut, dict):
        brut = brut.get("results", [])
    contenus = []
    for element in brut or []:
        if isinstance(element, str):
            contenus.append(element)
        elif isinstance(element, dict):
            contenus.append(element.get("memory") or element.get("text") or "")
    return [c for c in contenus if c]
