from agents_ollama.orchestrator_agent import create_orchestrator_agent
from agents_ollama.programmer_agent import create_programmer_agent
from agents_ollama.crypto_agent import create_crypto_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START

def create_graph():
    print("\nInitializing graph creation...")
    # Initialize agents
    programmer_agent = create_programmer_agent()
    crypto_agent = create_crypto_agent()
    print("✓ Agents initialized successfully")
    
    # Create workflow
    workflow = StateGraph(dict)
    print("✓ StateGraph created")
    
    # Add nodes
    print("\nAdding nodes to graph...")
    workflow.add_node("orchestrator", 
        lambda state: agent_node(state, create_orchestrator_agent(), "orchestrator"))
    workflow.add_node("programmer", 
        lambda state: agent_node(state, programmer_agent, "programmer"))
    workflow.add_node("crypto", 
        lambda state: agent_node(state, crypto_agent, "crypto"))
    print("✓ Nodes added successfully")

    # Setup edges
    print("\nConfiguring edges...")
    members = ["programmer", "crypto"]
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: x["next"] if "next" in x else "FINISH",
        conditional_map
    )
    print("✓ Edges configured successfully")
    
    print("\nCompiling graph...")
    return workflow.compile()

def agent_node(state, agent, name):
    print(f"\n[{name.upper()} NODE]")
    message = state["message"]
    current_messages = state.get("messages", [])
    
    print(f"Processing message: {message}")
    
    # Add message to the state for agents
    invoke_state = {
        "messages": current_messages + [HumanMessage(content=message)]
    }
    
    print(f"Invoking {name} agent...")
    result = agent.invoke(invoke_state)
    print(f"Agent result: {result}")
    
    # Update state
    updated_state = state.copy()
    if name != "orchestrator":
        output_content = result.get("output", "No output provided by agent.")
        updated_state["messages"] = current_messages + [
            SystemMessage(content=output_content, name=name)
        ]
        updated_state["final_response"] = output_content
        print(f"Final response set from {name}: {output_content[:100]}...")
    else:
        updated_state["next"] = result.get("next", "FINISH")
        print(f"Orchestrator decision: Next node will be '{updated_state['next']}'")
    
    return updated_state

if __name__ == "__main__":
    print("\n=== Starting Multi-Agent System ===")
    try:
        # Create the graph
        print("\nCreating agent graph...")
        graph = create_graph()
        print("✓ Graph created successfully")
        
        # Example message to test
        test_message = "Give me information about cryptocurrencies"
        print(f"\nTest message: '{test_message}'")
        
        # Create initial state
        initial_state = {
            "message": test_message,
            "messages": [],
            "final_response": "",
            "next": ""
        }
        print("✓ Initial state created")
        
        # Run the graph
        print("\nExecuting graph...")
        final_state = graph.invoke(initial_state)
        print("✓ Graph execution completed")
        
        # Print result
        print("\n=== FINAL RESULTS ===")
        print("Original message:", test_message)
        print("\nFinal response:", final_state.get("final_response"))
        print("\n=== Execution Complete ===")
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")