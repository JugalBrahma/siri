import os
# pyrefly: ignore [missing-import]
from langchain_tavily import TavilySearch

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

search_tool=TavilySearch(api_key=TAVILY_API_KEY)