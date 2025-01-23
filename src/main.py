from agents_ollama.orchestrator_agent import create_orchestrator_agent
from agents_ollama.programmer_agent import create_programmer_agent
from agents_ollama.crypto_agent import create_crypto_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
import sqlite3
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AgentState:
    message: str
    next: Optional[str] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}

# Mock responses for testing
MOCK_RESPONSES = {
    "orchestrator": {"next": "crypto"},
    "programmer": {"output": "This is a mock programmer response"},
    "crypto": {"output": "This is a mock cryptocurrency information response"}
}

class ChatSession:
    def __init__(self, user_id, db_path="sqlite:///chat_history.db"):
        try:
            self.user_id = user_id
            self.message_history = SQLChatMessageHistory(
                session_id=str(user_id),
                connection=db_path
            )
        except Exception as e:
            print(f"Error initializing ChatSession: {str(e)}")
            raise

    def add_message(self, message, is_human=True):
        try:
            if is_human:
                self.message_history.add_user_message(message)
            else:
                self.message_history.add_ai_message(message)
        except Exception as e:
            print(f"Error adding message: {str(e)}")
            raise

    @property
    def context(self):
        try:
            return "\n".join(str(message) for message in self.message_history.messages)
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            raise

def create_graph(chat_session):
    try:
        workflow = StateGraph(AgentState)
        
        orchestrator = create_orchestrator_agent()
        programmer = create_programmer_agent()
        crypto = create_crypto_agent()
        
        workflow.add_node("orchestrator", 
            lambda state: agent_node(state, orchestrator, "orchestrator", chat_session))
        workflow.add_node("programmer", 
            lambda state: agent_node(state, programmer, "programmer", chat_session))
        workflow.add_node("crypto", 
            lambda state: agent_node(state, crypto, "crypto", chat_session))
        
        workflow.add_edge(START, "orchestrator")
        workflow.add_conditional_edges(
            "orchestrator",
            lambda x: x.next if x.next else "FINISH",
            {"programmer": "programmer", "crypto": "crypto", "FINISH": END}
        )
        
        return workflow.compile()
    except Exception as e:
        print(f"Error creating graph: {str(e)}")
        raise

def agent_node(state: AgentState, agent, name: str, chat_session: ChatSession) -> AgentState:
    try:
        print(f"\n[{name.upper()} NODE]")

        message = state.message
        print(f"Processing message for user {chat_session.user_id}: {message}")
        history_messages = chat_session.message_history.messages
        
        if name != "orchestrator":
            chat_session.add_message(message, is_human=True)
        
        # Get mock response (for testing)
        # result = MOCK_RESPONSES[name]
        
        # Actual agent invocation
        result = agent.invoke({
            "messages": history_messages + [HumanMessage(content=message)]
        })
        
        if name != "orchestrator":
            output_content = result.get("output", "No output provided by agent.")
            chat_session.add_message(output_content, is_human=False)
            return AgentState(message=output_content)
        else:
            return AgentState(message=message, next=result["next"])

    except Exception as e:
        print(f"Error in agent node {name}: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("\n=== Starting Multi-Agent System ===")
        
        user_id = 1
        input_message = "Como me llamo?"
        chat_session = ChatSession(user_id)
        
        graph = create_graph(chat_session)
        initial_state = AgentState(message=input_message)
        final_state = graph.invoke(initial_state)
        
        print("\n=== FINAL RESULTS ===")
        print(f'Final response: {final_state['message']}')
        print("Complete context:")
        print(chat_session.context)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        raise