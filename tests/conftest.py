import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["MOCK_MODE"] = "true"
os.environ["DATA_DIR"] = str(ROOT / "data")


@pytest.fixture()
def llm():
    from support_agent.llm import MockChatLLM

    return MockChatLLM()


@pytest.fixture()
def app(llm):
    from support_agent.graph import build_graph
    from support_agent import tools

    tools.reset_tools()
    return build_graph(llm, os.environ["DATA_DIR"])
