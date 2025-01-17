from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser
from utils.llm_config import get_llm

def get_languages():
    return [
        {
            "name": "Python",
            "description": "High-level, interpreted, general-purpose programming language",
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
    languages_data = get_languages()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert programmer. Your task is to provide the following programming language information without modifying it:

            {languages_data}

            IMPORTANT RULES:
            - Return EXACTLY the JSON shown above
            - DO NOT add additional information
            - DO NOT modify the information
            - DO NOT make interpretations
            - DO NOT give advice
            - ONLY return the JSON exactly as it is

            REQUIRED RESPONSE FORMAT:
            You must return exactly the provided JSON, without adding or removing anything.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(languages_data=languages_data)

    def format_response(response: str) -> dict:
        return {"output": response}

    chain = prompt | get_llm() | format_response

    return chain