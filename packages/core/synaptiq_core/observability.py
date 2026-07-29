"""SynaptiQ — journalisation structurée et corrélation de requêtes.

## Pourquoi ce module existe

Trois constats de l'audit du 28/07 :

1. **Les tracebacks étaient perdus.** Une vingtaine de gestionnaires faisaient
   `logger.error(f"... : {e}")`, ce qui ne conserve que le message de l'exception. Sur un
   `KeyError` ou un `psycopg2.ProgrammingError`, le message seul ne dit ni où ni pourquoi.
   Une panne en production était donc indiagnosticable sans reproduction locale.
2. **Les logs n'étaient pas requêtables.** Format texte avec interpolation : impossible de
   filtrer par agent, par tenant ou par requête sans expression régulière fragile.
3. **`trace_id` n'était corrélable avec rien.** Il était dérivé d'un horodatage à la seconde
   (donc partagé par les requêtes concurrentes) et n'apparaissait dans AUCUN log. Il était
   retourné au client, qui ne pouvait rien en faire.

Aucune dépendance ajoutée : `logging` de la bibliothèque standard suffit.

## Usage

    from synaptiq_core.observability import configure_logging, set_trace_id

    configure_logging("synaptiq-api")        # au démarrage du process
    set_trace_id(trace_id)                   # au début du traitement d'une requête

Tout log émis ensuite, y compris depuis `synaptiq_core`, porte le `trace_id` courant.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sys

# Contexte de requête. Un contextvar (et non une variable globale) parce que FastAPI sert
# les routes synchrones dans un pool de threads : une globale mélangerait les requêtes.
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("synaptiq_trace_id", default="-")

# Attributs internes de LogRecord à ne pas recopier dans le JSON.
_ATTRIBUTS_STANDARD = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info", "thread",
    "threadName", "taskName", "trace_id", "service",
}


def set_trace_id(trace_id: str) -> None:
    """Associe un identifiant de corrélation au contexte d'exécution courant."""
    _trace_id.set(trace_id or "-")


def get_trace_id() -> str:
    return _trace_id.get()


class TraceIdFilter(logging.Filter):
    """Injecte le `trace_id` courant (et le nom du service) dans chaque enregistrement."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        record.service = self.service
        return True


class JsonFormatter(logging.Formatter):
    """Formate un enregistrement en une ligne JSON.

    Le traceback est mis dans un champ `exception` distinct plutôt que concaténé au
    message : un collecteur peut ainsi indexer le message et conserver la pile complète
    sans que la ligne devienne multi-lignes (ce qui casse la plupart des parseurs).
    """

    def format(self, record: logging.LogRecord) -> str:
        charge = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", "synaptiq"),
            "trace_id": getattr(record, "trace_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            charge["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            charge["stack"] = self.formatStack(record.stack_info)
        # Champs métier passés via `extra={...}` (agent_id, event_id, count…).
        for cle, valeur in record.__dict__.items():
            if cle not in _ATTRIBUTS_STANDARD and not cle.startswith("_"):
                charge[cle] = valeur
        return json.dumps(charge, ensure_ascii=False, default=str)


def configure_logging(service: str, level: str | None = None, stream=None) -> None:
    """Configure la journalisation du process (idempotent).

    `LOG_FORMAT=json` (défaut) produit une ligne JSON par enregistrement ; `text` conserve
    le format lisible à l'œil, utile en développement local.
    `LOG_LEVEL` accepte les niveaux usuels (INFO par défaut).

    ⚠️ `stream` vaut `sys.stdout` par défaut (convention des conteneurs : les logs vont sur
    la sortie standard, le collecteur s'en charge). **Un process qui utilise stdout comme
    canal de protocole doit passer `sys.stderr`** — c'est le cas du serveur MCP en transport
    `stdio`, où stdout porte le JSON-RPC : y écrire une ligne de log corrompt la session.
    """
    niveau = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    format_choisi = os.getenv("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    if format_choisi == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(trace_id)s] %(name)s %(levelname)s: %(message)s"))
    handler.addFilter(TraceIdFilter(service))

    racine = logging.getLogger()
    # Remplacer les handlers existants : `logging.basicConfig` a pu en poser un, et deux
    # handlers signifient chaque ligne loguée deux fois.
    for existant in list(racine.handlers):
        racine.removeHandler(existant)
    racine.addHandler(handler)
    racine.setLevel(niveau)
