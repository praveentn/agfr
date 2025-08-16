# agentic/agents/local/web_search_server.py
from fastmcp import FastMCP
import json
import time
import requests
from typing import Dict, List, Any
import os

mcp = FastMCP("Web Search Server")

@mcp.tool()
def search(query: str, limit: int = 8, recency_days: int = None) -> Dict[str, Any]:
    """Search the web for information using DuckDuckGo"""
    try:
        # Simple mock search results for demonstration
        # In production, integrate with actual search APIs
        mock_results = [
            {
                "title": f"Search result for '{query}' - Article {i+1}",
                "url": f"https://example.com/article-{i+1}",
                "snippet": f"This is a sample snippet for search result {i+1} about {query}. It contains relevant information and key insights.",
                "date_published": "2024-01-15"
            }
            for i in range(min(limit, 5))
        ]
        
        # Add some real-looking results for shoe market research
        if "shoe" in query.lower() or "market" in query.lower():
            mock_results = [
                {
                    "title": "Global Footwear Market Size, Share & Trends Analysis 2024",
                    "url": "https://www.grandviewresearch.com/industry-analysis/footwear-market",
                    "snippet": "The global footwear market size was valued at USD 365.5 billion in 2023 and is expected to grow at a CAGR of 5.5% from 2024 to 2030.",
                    "date_published": "2024-01-10"
                },
                {
                    "title": "Top Athletic Shoe Brands Leading Market Share in 2024",
                    "url": "https://www.marketresearch.com/athletic-shoes-2024",
                    "snippet": "Nike maintains 27% market share, followed by Adidas at 16%, and emerging brands gaining ground in sustainable footwear.",
                    "date_published": "2024-01-12"
                },
                {
                    "title": "Consumer Preferences in Footwear: Comfort vs Style Trends",
                    "url": "https://www.retailnews.com/shoe-consumer-trends-2024", 
                    "snippet": "Recent surveys show 68% of consumers prioritize comfort over style, driving innovation in cushioning technology and ergonomic design.",
                    "date_published": "2024-01-08"
                }
            ]
        
        return {
            "items": mock_results[:limit],
            "total_found": len(mock_results),
            "query": query,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "items": [],
            "error": str(e),
            "query": query
        }

if __name__ == "__main__":
    print("Starting Web Search Server on port 9101...")
    mcp.run(transport="sse", host="0.0.0.0", port=9101)


