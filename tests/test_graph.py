from src.main import create_graph

def test_orchestrator_routing(mock_chat_session):
    graph = create_graph(mock_chat_session)
    
    # Test crypto query
    crypto_state = graph.invoke({"message": "Tell me about Bitcoin"})
    assert crypto_state["message"] is not None
    assert len(crypto_state["message"]) > 0
    
    # Test programming query
    prog_state = graph.invoke({"message": "Help me with Python code"})
    assert prog_state["message"] is not None
    assert len(prog_state["message"]) > 0