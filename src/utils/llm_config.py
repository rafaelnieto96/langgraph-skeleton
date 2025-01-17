from langchain_ollama import OllamaLLM

def get_llm():
    return OllamaLLM(
        model="mixtral:8x7b",
        temperature=0
    )
