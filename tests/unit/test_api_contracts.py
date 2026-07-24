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
        ContextConstraints(memory_types=["unknown"])


def test_memory_input_rejects_invalid_score_and_type():
    with pytest.raises(ValidationError):
        MemoryInput(agent_id="agent", type="unknown", content="memory")
    with pytest.raises(ValidationError):
        MemoryInput(agent_id="agent", type="semantic", content="memory", importance=1.1)


def test_retrieve_request_limit_is_bounded():
    with pytest.raises(ValidationError):
        RetrieveRequest(agent_id="agent", query="memory", limit=101)
