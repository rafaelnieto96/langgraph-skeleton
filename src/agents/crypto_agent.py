import json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from utils.llm_config import get_llm

llm = get_llm()

@tool
def get_cryptocurrencies() -> str:
    """Gets information about cryptocurrencies."""
    return [
        {
            "name": "Bitcoin",
            "description": "The first and most well-known cryptocurrency, based on blockchain technology",
            "current_price": "66,245.00 USD"
        },
        {
            "name": "Ethereum",
            "description": "Decentralized platform that enables smart contracts and DApps",
            "current_price": "3,284.75 USD"
        },
        {
            "name": "Solana",
            "description": "High-speed blockchain for DeFi and NFTs",
            "current_price": "128.35 USD"
        }
    ]

def create_crypto_agent():
    tools = [
        Tool(
            name="GetCryptocurrencies",
            func=get_cryptocurrencies,
            description="Gets updated information about cryptocurrencies"
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert analyst in cryptocurrencies and digital markets.

            Your task is to provide updated information about cryptocurrencies. You MUST:

            1. ALWAYS use the GetCryptocurrencies tool to obtain information
            2. Return the response in JSON format without modifying the obtained information

            IMPORTANT RULES:
            - ALWAYS use the GetCryptocurrencies tool
            - DO NOT add additional information outside the JSON
            - DO NOT modify the obtained information
            - DO NOT make summaries or interpretations
            - DO NOT add investment predictions or advice
            - DO NOT use your general knowledge
            - Return EXACTLY the JSON you get from the tool

            RESPONSE EXAMPLE:
            [
                {{
                    "name": "Bitcoin",
                    "description": "The first and most well-known cryptocurrency...",
                    "current_price": "66,245.00 USD"
                }},
                ...
            ]
            """),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return executor