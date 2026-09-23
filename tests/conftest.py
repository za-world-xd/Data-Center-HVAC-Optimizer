from pathlib import Path

import pytest

from engine.schemas import Scenario


@pytest.fixture
def scenario():
    return Scenario.model_validate_json(Path("examples/scenario.json").read_text())
