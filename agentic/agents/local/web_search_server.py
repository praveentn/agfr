# agentic/agents/local/web_search_server.py
import asyncio
import time
from typing import Dict, Any

from fastmcp import FastMCP

mcp = FastMCP("Web Search Server")

@mcp.tool()
def search(query: str, limit: int = 8, recency_days: int = None) -> Dict[str, Any]:
    """
    Search the web for information (mock implementation).
    Replace with actual search API integration in production.
    """
    try:
        # Default mock results
        mock_results = [
            {
                "title": f"Search result for '{query}' - Article {i+1}",
                "url": f"https://example.com/article-{i+1}",
                "snippet": f"Sample snippet for result {i+1} about {query}.",
                "date_published": "2024-01-15",
            }
            for i in range(min(limit, 5))
        ]

        # Special handling for "shoe market" queries
        if "shoe" in query.lower() or "market" in query.lower():
            mock_results = [
                {
                    "title": "Global Footwear Market Size, Share & Trends Analysis 2024",
                    "url": "https://www.grandviewresearch.com/industry-analysis/footwear-market",
                    "snippet": "The global footwear market size was valued at USD 365.5B in 2023 and is expected to grow at a CAGR of 5.5% through 2030.",
                    "date_published": "2024-01-10",
                },
                {
                    "title": "Top Athletic Shoe Brands Leading Market Share in 2024",
                    "url": "https://www.marketresearch.com/athletic-shoes-2024",
                    "snippet": "Nike holds 27% market share, Adidas 16%, with sustainable footwear brands gaining traction.",
                    "date_published": "2024-01-12",
                },
                {
                    "title": "Consumer Preferences in Footwear: Comfort vs Style Trends",
                    "url": "https://www.retailnews.com/shoe-consumer-trends-2024",
                    "snippet": "68% of consumers now prioritize comfort over style, fueling ergonomic design innovations.",
                    "date_published": "2024-01-08",
                },
            ]

        return {
            "items": mock_results[:limit],
            "total_found": len(mock_results),
            "query": query,
            "timestamp": time.time(),
        }

    except Exception as e:
        return {
            "items": [],
            "error": str(e),
            "query": query,
        }

async def main():
    print("Starting Web Search Server on port 9101 (async mode)...")
    await mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=9101,
        log_level="info"
    )

if __name__ == "__main__":
    asyncio.run(main())
