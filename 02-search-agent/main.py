import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient


load_dotenv("../.env")

ollama_model = os.getenv("OLLAMA_MODEL")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """
    A tool that searches the internet
    Args:
        query (str): The query to search for
    Returns:
        The result of the search
    """
    print(f"Searching {query}")
    return tavily_client.search(query)


llm = ChatOllama(endpoint=ollama_base_url, model=ollama_model)
tools = [search]
agent = create_agent(llm, tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo?")}
    )
    print(result)


if __name__ == "__main__":
    main()
