from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from utils.llm_config import get_llm

def create_orchestrator_agent():
    response_schemas = [
        ResponseSchema(name="next", 
                      description="The next worker that should act",
                      type="string")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    system_prompt = """
    You are a supervisor in charge of managing a conversation between the following workers: {members}.
    You must decide which worker will act next.

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

    REQUIRED FORMAT:
    You must respond only with a JSON with this structure:
    {{"next": "worker_name"}}
    
    Where worker_name must be "programmer" or "crypto"
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Given the previous conversation context, who should act next?")
    ]).partial(members="programmer, crypto")

    llm = get_llm()

    chain = prompt | llm | output_parser

    return chain