import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "core"))

from apps.api.main import ContextConstraints, EventInput, MemoryInput, RetrieveRequest


def test_event_input_rejects_unbounded_agent_id():
    with pytest.raises(ValidationError):
        EventInput(agent_id="x" * 51, session_id="s", content="content")


def test_context_constraints_reject_invalid_budget_and_type():
    with pytest.raises(ValidationError):
        ContextConstraints(max_tokens=0)
    with pytest.raises(ValidationError):
        ContextConstraints(compartments=["unknown"])


def test_memory_input_rejects_invalid_score():
    with pytest.raises(ValidationError):
        MemoryInput(agent_id="agent", content="memory", importance=1.1)


def test_memory_input_compartment_inconnu_accepte_pour_classement():
    """À l'écriture, un nom inconnu n'est pas une erreur : le moteur classera."""
    assert MemoryInput(agent_id="agent", content="m", compartment="truc").compartment == "truc"


def test_memory_input_indices_reserves_aux_croyances():
    with pytest.raises(ValidationError):
        MemoryInput(agent_id="agent", content="m", compartment="savoirs", evidence=["x"])


def test_memory_input_ancienne_forme_de_croyance_convertie():
    m = MemoryInput(agent_id="a", content="Jimmy aime X", type="reflective",
                    subtype="human_insights", confidence=0.5)
    assert (m.compartment, m.about) == ("croyances", "humans")


def test_retrieve_request_limit_is_bounded():
    with pytest.raises(ValidationError):
        RetrieveRequest(agent_id="agent", query="memory", limit=101)
