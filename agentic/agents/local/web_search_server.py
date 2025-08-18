# agentic/agents/local/web_search_server.py
import time
import sys
import os
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastmcp import FastMCP
from agentic.core.config import settings

# Import search engine if available
try:
    from modules.search_engine import WebSearchEngine
    search_engine = WebSearchEngine()
    SEARCH_AVAILABLE = True
except ImportError:
    print("Warning: Search engine module not available, using fallback")
    search_engine = None
    SEARCH_AVAILABLE = False

mcp = FastMCP("Web Search Server")

@mcp.tool()
def search(query: str, limit: int = 8, recency_days: int = None) -> Dict[str, Any]:
    """
    Search the web for information using integrated search engine.
    Falls back to mock data if search engine unavailable.
    """
    try:
        if SEARCH_AVAILABLE and search_engine:
            # Use real search engine
            search_type = "general"
            if any(keyword in query.lower() for keyword in ["market", "industry", "competitive", "analysis"]):
                search_type = "company"
            elif any(keyword in query.lower() for keyword in ["startup", "funding", "venture"]):
                search_type = "startup"
            elif any(keyword in query.lower() for keyword in ["technology", "tech", "innovation"]):
                search_type = "technology"
            
            print(f"Performing real search for: {query} (type: {search_type})")
            results = search_engine.comprehensive_search(query, search_type, limit)
            
            # Transform results to expected format
            items = []
            for result in results:
                items.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "date_published": result.get("date_published", ""),
                    "domain": result.get("domain", ""),
                    "relevance_score": result.get("relevance_score", 0.5)
                })
            
            return {
                "items": items,
                "total_found": len(items),
                "query": query,
                "timestamp": time.time(),
                "source": "real_search"
            }
        else:
            # Fallback to enhanced mock results
            return _get_enhanced_mock_results(query, limit)

    except Exception as e:
        print(f"Search failed: {e}")
        return _get_enhanced_mock_results(query, limit)

def _get_enhanced_mock_results(query: str, limit: int) -> Dict[str, Any]:
    """Enhanced mock results based on query content"""
    mock_results = []
    
    # Generate contextual mock results based on query keywords
    if "shoe" in query.lower() or "footwear" in query.lower():
        mock_results = [
            {
                "title": "Global Footwear Market Size, Share & Trends Analysis 2024",
                "url": "https://www.grandviewresearch.com/industry-analysis/footwear-market",
                "snippet": "The global footwear market size was valued at USD 365.5B in 2023 and is expected to grow at a CAGR of 5.5% through 2030. Athletic footwear dominates with 40% market share.",
                "date_published": "2024-01-10",
                "domain": "grandviewresearch.com",
                "relevance_score": 0.9
            },
            {
                "title": "Top Athletic Shoe Brands Leading Market Share in 2024",
                "url": "https://www.marketresearch.com/athletic-shoes-2024",
                "snippet": "Nike holds 27% market share, Adidas 16%, with sustainable footwear brands gaining traction. Direct-to-consumer sales up 15% year-over-year.",
                "date_published": "2024-01-12",
                "domain": "marketresearch.com",
                "relevance_score": 0.85
            },
            {
                "title": "Consumer Preferences in Footwear: Comfort vs Style Trends",
                "url": "https://www.retailnews.com/shoe-consumer-trends-2024",
                "snippet": "68% of consumers now prioritize comfort over style, fueling ergonomic design innovations. Price sensitivity increased 12% post-pandemic.",
                "date_published": "2024-01-08",
                "domain": "retailnews.com",
                "relevance_score": 0.8
            }
        ]
    elif "electric vehicle" in query.lower() or "ev" in query.lower():
        mock_results = [
            {
                "title": "Electric Vehicle Market Growth Projections 2024-2030",
                "url": "https://www.evmarket-analysis.com/global-trends-2024",
                "snippet": "Global EV market expected to reach $1.7 trillion by 2030. Tesla maintains 18% market share while Chinese manufacturers gain ground.",
                "date_published": "2024-01-15",
                "domain": "evmarket-analysis.com",
                "relevance_score": 0.95
            },
            {
                "title": "Battery Technology Breakthroughs Driving EV Adoption",
                "url": "https://www.tech-innovations.com/ev-battery-2024",
                "snippet": "Solid-state batteries promise 50% more range and 10-minute charging. Major automakers investing $100B in battery tech.",
                "date_published": "2024-01-12",
                "domain": "tech-innovations.com",
                "relevance_score": 0.9
            },
            {
                "title": "Government EV Incentives and Infrastructure Investment",
                "url": "https://www.policy-tracker.com/ev-incentives-2024",
                "snippet": "Global governments allocate $500B for EV infrastructure. Federal tax credits extended through 2025 in major markets.",
                "date_published": "2024-01-10",
                "domain": "policy-tracker.com",
                "relevance_score": 0.85
            }
        ]
    elif "market research" in query.lower() or "industry analysis" in query.lower():
        topic = query.replace("market research", "").replace("industry analysis", "").strip()
        mock_results = [
            {
                "title": f"{topic.title()} Market Analysis & Industry Report 2024",
                "url": f"https://www.industry-reports.com/{topic.lower().replace(' ', '-')}-market-2024",
                "snippet": f"Comprehensive analysis of the {topic} market including size, growth projections, competitive landscape, and emerging trends shaping the industry.",
                "date_published": "2024-01-15",
                "domain": "industry-reports.com",
                "relevance_score": 0.9
            },
            {
                "title": f"Top Companies in {topic.title()} Industry - 2024 Rankings",
                "url": f"https://www.business-insights.com/{topic.lower().replace(' ', '-')}-companies-2024",
                "snippet": f"Leading players in the {topic} space, market share analysis, recent developments, and strategic initiatives driving growth.",
                "date_published": "2024-01-12",
                "domain": "business-insights.com",
                "relevance_score": 0.85
            },
            {
                "title": f"{topic.title()} Industry Trends and Future Outlook",
                "url": f"https://www.trend-analysis.com/{topic.lower().replace(' ', '-')}-trends-2024",
                "snippet": f"Key trends shaping the {topic} industry including technological innovations, regulatory changes, and consumer behavior shifts.",
                "date_published": "2024-01-10",
                "domain": "trend-analysis.com",
                "relevance_score": 0.8
            }
        ]
    else:
        # Generic results
        mock_results = [
            {
                "title": f"Comprehensive Guide to {query.title()}",
                "url": f"https://www.comprehensive-guides.com/{query.lower().replace(' ', '-')}",
                "snippet": f"Complete overview and analysis of {query}, including latest developments, key insights, and expert perspectives.",
                "date_published": "2024-01-15",
                "domain": "comprehensive-guides.com",
                "relevance_score": 0.7
            },
            {
                "title": f"{query.title()} - Latest News and Updates",
                "url": f"https://www.news-source.com/{query.lower().replace(' ', '-')}-news",
                "snippet": f"Stay updated with the latest news, trends, and developments related to {query}.",
                "date_published": "2024-01-12",
                "domain": "news-source.com",
                "relevance_score": 0.6
            }
        ]
    
    return {
        "items": mock_results[:limit],
        "total_found": len(mock_results),
        "query": query,
        "timestamp": time.time(),
        "source": "mock_search"
    }

if __name__ == "__main__":
    print("Starting Web Search Server on port 9101...")
    print(f"Search engine available: {SEARCH_AVAILABLE}")
    
    # Use sync run to avoid asyncio conflicts
    mcp.run(transport="http", host="0.0.0.0", port=9101)
