"""Tests unitaires de la journalisation structurée (audit F14).

Ce qui doit être vrai pour que les logs servent en production : le traceback est conservé,
le `trace_id` est présent sur chaque ligne, et les champs métier passés via `extra=` se
retrouvent dans le JSON (sinon aucun filtrage par agent ou par tenant n'est possible).
"""
import json
import logging

import pytest

from synaptiq_core.observability import (
    JsonFormatter,
    TraceIdFilter,
    configure_logging,
    get_trace_id,
    set_trace_id,
)


@pytest.fixture(autouse=True)
def _trace_neutre():
    set_trace_id("-")
    yield
    set_trace_id("-")


def _formater(record: logging.LogRecord, service: str = "test-service") -> dict:
    TraceIdFilter(service).filter(record)
    return json.loads(JsonFormatter().format(record))


def _record(message="un message", niveau=logging.INFO, **extra):
    record = logging.LogRecord(name="synaptiq-test", level=niveau, pathname=__file__,
                              lineno=10, msg=message, args=(), exc_info=None)
    for cle, valeur in extra.items():
        setattr(record, cle, valeur)
    return record


def test_ligne_json_avec_les_champs_attendus():
    charge = _formater(_record("mémoire créée"))
    assert charge["message"] == "mémoire créée"
    assert charge["level"] == "INFO"
    assert charge["logger"] == "synaptiq-test"
    assert charge["service"] == "test-service"
    assert "ts" in charge


def test_le_trace_id_courant_est_injecte():
    set_trace_id("trace_abcdef")
    assert get_trace_id() == "trace_abcdef"
    assert _formater(_record())["trace_id"] == "trace_abcdef"


def test_sans_trace_id_valeur_neutre():
    assert _formater(_record())["trace_id"] == "-"


def test_le_traceback_est_conserve_dans_un_champ_dedie():
    """Le cœur de F14 : `logger.error(f"...{e}")` perdait entièrement la pile."""
    try:
        {}["absente"]
    except KeyError:
        import sys
        record = logging.LogRecord(name="synaptiq-test", level=logging.ERROR,
                                   pathname=__file__, lineno=20, msg="échec",
                                   args=(), exc_info=sys.exc_info())
    charge = _formater(record)
    assert charge["message"] == "échec"
    assert "KeyError" in charge["exception"]
    assert "Traceback" in charge["exception"]
    # Le message reste indexable seul : la pile ne le pollue pas.
    assert "Traceback" not in charge["message"]


def test_les_champs_metier_sont_exposes():
    charge = _formater(_record("consolidation", agent_id="agentA", event_id="evt-1", count=3))
    assert charge["agent_id"] == "agentA"
    assert charge["event_id"] == "evt-1"
    assert charge["count"] == 3


def test_json_reste_sur_une_seule_ligne():
    """Un log multi-lignes casse la plupart des collecteurs."""
    try:
        raise ValueError("erreur sur\nplusieurs lignes")
    except ValueError:
        import sys
        record = logging.LogRecord(name="t", level=logging.ERROR, pathname=__file__,
                                   lineno=1, msg="msg\navec saut", args=(),
                                   exc_info=sys.exc_info())
    rendu = JsonFormatter().format(record)
    assert "\n" not in rendu
    assert json.loads(rendu)["message"] == "msg\navec saut"


def test_objet_non_serialisable_ne_fait_pas_echouer_le_log():
    """Un log qui lève masquerait l'incident qu'il devait signaler."""
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    charge = _formater(_record("etat", objet=Opaque()))
    assert charge["objet"] == "<opaque>"


def test_configure_logging_ne_duplique_pas_les_handlers(monkeypatch):
    """Appelé deux fois (import multiple, rechargement), il ne doit pas doubler les lignes."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging("service-a")
    configure_logging("service-b")
    assert len(logging.getLogger().handlers) == 1


def test_format_texte_disponible_pour_le_developpement(monkeypatch, capsys):
    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_logging("service-texte")
    set_trace_id("trace_xyz")
    logging.getLogger("synaptiq-test").info("message lisible")
    sortie = capsys.readouterr().out
    assert "message lisible" in sortie
    assert "trace_xyz" in sortie
    assert not sortie.strip().startswith("{")
