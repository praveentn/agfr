# agentic/agents/local/nlp_summarizer_server.py
import asyncio
import re
import sys
import os
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastmcp import FastMCP
from agentic.core.llm_client import llm_client

mcp = FastMCP("NLP Summarizer Server")

@mcp.tool()
async def summarize(text: str, max_length: int = 200, style: str = "brief") -> Dict[str, Any]:
    """
    Summarize text content using Azure OpenAI with fallback to local processing.
    Supports 'brief', 'detailed', and 'bullet_points' styles.
    """
    try:
        # Try Azure OpenAI first
        if llm_client and llm_client.client:
            print(f"Using Azure OpenAI for summarization (style: {style})")
            result = await llm_client.summarize_text(text, style, max_length)
            if result and result.get("summary"):
                return result
        
        # Fallback to local processing
        print("Using local NLP processing for summarization")
        return _local_summarize(text, max_length, style)

    except Exception as e:
        print(f"Summarization failed: {e}")
        return _local_summarize(text, max_length, style)

def _local_summarize(text: str, max_length: int, style: str) -> Dict[str, Any]:
    """Local text summarization fallback"""
    try:
        cleaned = re.sub(r'\s+', ' ', text.strip())
        sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]
        key_points = _extract_key_points(sentences)

        if style == "bullet_points":
            summary = "\n".join(f"• {pt}" for pt in key_points[:5])
        elif style == "detailed":
            important = _select_important_sentences(sentences, max(3, len(sentences) // 3))
            summary = ". ".join(important)
        else:  # brief
            important = _select_important_sentences(sentences, min(3, len(sentences)))
            summary = ". ".join(important)

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return {
            "summary": summary,
            "key_points": key_points,
            "word_count": len(summary.split()),
            "original_length": len(text),
            "compression_ratio": round(len(summary) / len(text), 2) if text else 0,
            "source": "local_processing"
        }

    except Exception as e:
        return {
            "summary": f"Error generating summary: {str(e)}",
            "key_points": [],
            "error": str(e),
            "word_count": 0,
            "source": "error"
        }

def _extract_key_points(sentences: list) -> list:
    """Extract key points from sentences"""
    keywords = [
        "market", "trend", "growth", "increase", "decrease", "analysis", 
        "research", "study", "report", "data", "statistics", "percent",
        "million", "billion", "company", "industry", "consumer", "customer",
        "revenue", "profit", "share", "competition", "technology", "innovation"
    ]
    points = []
    for s in sentences:
        if any(k in s.lower() for k in keywords) and 20 < len(s) < 200:
            points.append(s)
        if len(points) >= 8:
            break
    return points

def _select_important_sentences(sentences: list, num_sentences: int) -> list:
    """Select most important sentences based on scoring"""
    if len(sentences) <= num_sentences:
        return sentences

    scored = []
    for idx, s in enumerate(sentences):
        words = s.lower().split()
        score = 0
        
        # Length scoring (prefer medium length sentences)
        score += 2 if 10 <= len(words) <= 25 else 1 if 5 <= len(words) < 10 or 25 < len(words) <= 35 else 0
        
        # Keyword scoring
        score += sum(1 for w in words if w in [
            "market", "research", "analysis", "trend", "data", "study", 
            "report", "industry", "company", "growth", "revenue", "share"
        ])
        
        # Position bonus (earlier sentences get higher score)
        pos_bonus = max(0, 2 - (idx / len(sentences)) * 2)
        score += pos_bonus
        
        scored.append((s, score))

    # Select top scored sentences and maintain original order
    selected = [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:num_sentences]]
    result = [s for s in sentences if s in selected]
    return result[:num_sentences]

@mcp.tool()
async def extract_entities(text: str) -> Dict[str, Any]:
    """Extract named entities from text"""
    try:
        # Simple regex-based entity extraction as fallback
        entities = {
            "organizations": [],
            "locations": [],
            "numbers": [],
            "dates": []
        }
        
        # Extract organizations (capitalized words followed by Corp, Inc, Ltd, etc.)
        org_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Corp|Inc|Ltd|LLC|Company|Industries|Group|Holdings)\b'
        entities["organizations"] = list(set(re.findall(org_pattern, text)))
        
        # Extract locations (capitalized words that might be places)
        location_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?:\s(?:City|State|Country|County|Province))\b'
        entities["locations"] = list(set(re.findall(location_pattern, text)))
        
        # Extract numbers with units
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|percent|%|USD|dollars?))\b'
        entities["numbers"] = list(set(re.findall(number_pattern, text, re.IGNORECASE)))
        
        # Extract dates
        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\w+\s\d{1,2},?\s\d{4})\b'
        entities["dates"] = list(set(re.findall(date_pattern, text)))
        
        return {
            "entities": entities,
            "total_entities": sum(len(v) for v in entities.values()),
            "source": "regex_extraction"
        }
        
    except Exception as e:
        return {
            "entities": {"organizations": [], "locations": [], "numbers": [], "dates": []},
            "error": str(e),
            "source": "error"
        }

if __name__ == "__main__":
    print("Starting NLP Summarizer Server on port 9103...")
    print(f"Azure OpenAI client available: {llm_client and llm_client.client is not None}")
    
    # Use sync run to avoid asyncio conflicts
    mcp.run(transport="http", host="0.0.0.0", port=9103)
