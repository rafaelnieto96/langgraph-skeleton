import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

import pytest
from agents_ollama.orchestrator_agent import create_orchestrator_agent
from agents_ollama.programmer_agent import create_programmer_agent
from agents_ollama.crypto_agent import create_crypto_agent
from main import ChatSession

@pytest.fixture
def mock_chat_session():
    return ChatSession(user_id=1, db_path="sqlite:///test_chat_history.db")

@pytest.fixture
def mock_agents():
    return {
        "orchestrator": create_orchestrator_agent(),
        "programmer": create_programmer_agent(),
        "crypto": create_crypto_agent()
    }