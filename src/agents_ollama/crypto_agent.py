from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser
from utils.llm_config import get_llm

def get_cryptocurrencies():
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
    crypto_data = get_cryptocurrencies()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert cryptocurrency analyst. Your task is to provide the following cryptocurrency information without modifying it:

            {crypto_data}

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
    ]).partial(crypto_data=crypto_data)

    def format_response(response: str) -> dict:
        return {"output": response}

    chain = prompt | get_llm() | format_response

    return chain