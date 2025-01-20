from agents_ollama.orchestrator_agent import create_orchestrator_agent
from agents_ollama.programmer_agent import create_programmer_agent
from agents_ollama.crypto_agent import create_crypto_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
import sqlite3
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Mock responses for testing
MOCK_RESPONSES = {
    "orchestrator": {"next": "crypto"},
    "programmer": {"output": "This is a mock programmer response"},
    "crypto": {"output": "This is a mock cryptocurrency information response"}
}

class ChatSession:
    def __init__(self, user_id, db_path="sqlite:///chat_history.db"):
        self.user_id = user_id
        self.message_history = SQLChatMessageHistory(
            session_id=str(user_id),
            connection_string=db_path
        )

    def add_message(self, message, is_human=True):
        if is_human:
            self.message_history.add_user_message(message)
        else:
            self.message_history.add_ai_message(message)

    @property
    def context(self):
        return "\n".join(str(message) for message in self.message_history.messages)

def create_graph(chat_session):
    print("\nInitializing graph creation...")
    workflow = StateGraph(dict)
    
    orchestrator = create_orchestrator_agent()
    programmer = create_programmer_agent()
    crypto = create_crypto_agent()
    
    print("\nAdding nodes to graph...")
    workflow.add_node("orchestrator", 
        lambda state: agent_node(state, orchestrator, "orchestrator", chat_session))
    workflow.add_node("programmer", 
        lambda state: agent_node(state, programmer, "programmer", chat_session))
    workflow.add_node("crypto", 
        lambda state: agent_node(state, crypto, "crypto", chat_session))
    
    print("\nConfiguring edges...")
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: x["next"] if "next" in x else "FINISH",
        {"programmer": "programmer", "crypto": "crypto", "FINISH": END}
    )
    
    return workflow.compile()

def agent_node(state, agent, name, chat_session):
    print(f"\n[{name.upper()} NODE]")
    message = state["message"]
    
    print(f"Processing message for user {chat_session.user_id}: {message}")
    history_messages = chat_session.message_history.messages
    
    if name != "orchestrator":
        chat_session.add_message(message, is_human=True)
    
    result = MOCK_RESPONSES[name]
    
    # result = agent.invoke({
    #     "messages": history_messages + [HumanMessage(content=message)]
    # })
    
    if name != "orchestrator":
        output_content = result.get("output", "No output provided by agent.")
        chat_session.add_message(output_content, is_human=False)
        return {
            "message": output_content
        }
    else:
        return {
            "message": message,
            "next": result["next"]
        }

if __name__ == "__main__":
    print("\n=== Starting Multi-Agent System ===")
    try:
        user_id = 1
        input_message = "Give me information about cryptocurrencies"
        chat_session = ChatSession(user_id)
        
        graph = create_graph(chat_session)
        
        initial_state = {
            "message": input_message
        }
        
        final_state = graph.invoke(initial_state)
        print('final_statefinal_statefinal_state', final_state)
        print("\n=== FINAL RESULTS ===")
        print(f"User ID: {user_id}")
        print("Complete context:")
        print(chat_session.context)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")