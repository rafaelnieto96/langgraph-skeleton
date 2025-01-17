from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.tools import tool
from typing import Literal
from utils.llm_config import get_llm

def create_orchestrator_agent():
    function_def = {
        "name": "route",
        "description": "Selects the next worker",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": ["programmer", "crypto"],
                    "description": "The next worker that should act"
                }
            },
            "required": ["next"]
        }
    }

    system_prompt = """
    You are a supervisor in charge of managing a conversation between the following workers: {members}.
    You MUST use the 'route' function to indicate which worker will act next.

    Worker functions:
    - programmer: Provides information about programming languages, their versions, and characteristics.
    - crypto: Provides information about cryptocurrencies and their current prices.

    DECISION RULES:
    For programmer:
    - When asking about programming languages
    - When mentioning Python, JavaScript, Java, or other languages
    - When asking about language versions
    - When looking for information about software development

    For crypto:
    - When asking about cryptocurrencies
    - When mentioning Bitcoin, Ethereum, or other cryptocurrencies
    - When asking about cryptocurrency prices
    - When looking for information about blockchain

    If you're not sure, select the worker that seems most appropriate based on the conversation context.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Given the previous conversation context, who should act next?")
    ]).partial(members="programmer, crypto")

    llm = get_llm()

    supervisor_chain = (
        prompt
        | llm.bind(
            functions=[function_def],
            function_call={"name": "route"}
        )
        | JsonOutputFunctionsParser()
    )

    return supervisor_chain