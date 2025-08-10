from langchain_tavily import TavilySearch
import os


def get_tavily_tool():
    """Get configured Tavily search tool.
    
    Returns a TavilySearch tool configured with:
    - max_results: 3 (keep responses concise)
    - Requires TAVILY_API_KEY in environment
    """
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    
    tool = TavilySearch(max_results=3)
    tool.name = "tavily_search"
    tool.description = "Search the web for current information, facts, and real-time data"
    return tool