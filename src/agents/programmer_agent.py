import json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from utils.llm_config import get_llm

llm = get_llm()

@tool
def get_languages() -> str:
    """Gets information about programming languages."""
    return [
        {
            "name": "Python",
            "description": "High-level, interpreted, general-purpose language",
            "latest_version": "3.12.1"
        },
        {
            "name": "JavaScript",
            "description": "Programming language for web and application development",
            "latest_version": "ES2024"
        },
        {
            "name": "Java",
            "description": "Object-oriented, compiled, cross-platform language",
            "latest_version": "JDK 21"
        }
    ]

def create_programmer_agent():
    tools = [
        Tool(
            name="GetLanguages",
            func=get_languages,
            description="Gets updated information about programming languages"
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert programmer with extensive knowledge in multiple programming languages.

            Your task is to provide updated information about programming languages. You MUST:

            1. ALWAYS use the GetLanguages tool to obtain the information
            2. Return the response in JSON format without modifying the obtained information

            IMPORTANT RULES:
            - ALWAYS use the GetLanguages tool
            - DO NOT add additional information outside the JSON
            - DO NOT modify the obtained information
            - DO NOT make summaries or interpretations
            - DO NOT use your general knowledge
            - Return EXACTLY the JSON you get from the tool

            RESPONSE EXAMPLE:
            [
                {{
                    "name": "Python",
                    "description": "High-level language...",
                    "latest_version": "3.12.1"
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