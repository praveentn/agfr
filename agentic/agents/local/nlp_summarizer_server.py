# agentic/agents/local/nlp_summarizer_server.py
import asyncio
import re
import sys
import os
import logging
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastmcp import FastMCP
from agentic.core.llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("NLP Summarizer Server")

@mcp.tool()
async def summarize(text: str, max_length: int = 200, style: str = "brief") -> Dict[str, Any]:
    """
    Summarize text content using Azure OpenAI with fallback to local processing.
    Supports 'brief', 'detailed', and 'bullet_points' styles.
    
    Args:
        text: Text to summarize (minimum 10 characters)
        max_length: Maximum summary length in characters (50-2000)
        style: Summary style - brief, detailed, or bullet_points
    
    Returns:
        Dictionary with summary, key points, and metadata
    """
    try:
        logger.info(f"Summarizing text (length: {len(text)}, style: {style})")
        
        # Validate inputs
        if len(text) < 10:
            raise ValueError("Text too short for meaningful summarization")
        
        if not 50 <= max_length <= 2000:
            max_length = max(50, min(2000, max_length))
        
        if style not in ["brief", "detailed", "bullet_points"]:
            style = "brief"
        
        # Try Azure OpenAI first
        if llm_client and llm_client.client:
            logger.info("Using Azure OpenAI for summarization")
            result = await llm_client.summarize_text(text, style, max_length)
            if result and result.get("summary"):
                result["source"] = "azure_openai"
                logger.info("Azure OpenAI summarization successful")
                return result
        
        # Fallback to local processing
        logger.info("Using local NLP processing for summarization")
        return _local_summarize(text, max_length, style)

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
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
            "source": "local_processing",
            "style": style,
            "success": True
        }

    except Exception as e:
        logger.error(f"Local summarization failed: {e}")
        return {
            "summary": f"Error generating summary: {str(e)}",
            "key_points": [],
            "error": str(e),
            "word_count": 0,
            "source": "error",
            "success": False
        }

def _extract_key_points(sentences: list) -> list:
    """Extract key points from sentences"""
    keywords = [
        "market", "trend", "growth", "increase", "decrease", "analysis", 
        "research", "study", "report", "data", "statistics", "percent",
        "million", "billion", "company", "industry", "consumer", "customer",
        "revenue", "profit", "share", "competition", "technology", "innovation",
        "important", "significant", "key", "major", "primary", "main"
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
            "report", "industry", "company", "growth", "revenue", "share",
            "important", "significant", "key", "major", "shows", "indicates"
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
    """
    Extract named entities from text (organizations, locations, numbers, dates).
    
    Args:
        text: Text to extract entities from (minimum 10 characters)
    
    Returns:
        Dictionary with extracted entities and metadata
    """
    try:
        logger.info(f"Extracting entities from text (length: {len(text)})")
        
        if len(text) < 10:
            raise ValueError("Text too short for entity extraction")
        
        # Simple regex-based entity extraction as fallback
        entities = {
            "organizations": [],
            "locations": [],
            "numbers": [],
            "dates": [],
            "people": []
        }
        
        # Extract organizations (capitalized words followed by Corp, Inc, Ltd, etc.)
        org_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Corp|Inc|Ltd|LLC|Company|Industries|Group|Holdings|Corporation)\b'
        entities["organizations"] = list(set(re.findall(org_pattern, text)))
        
        # Extract locations (capitalized words that might be places)
        location_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?:\s(?:City|State|Country|County|Province|Street|Avenue|Road))\b'
        entities["locations"] = list(set(re.findall(location_pattern, text)))
        
        # Extract numbers with units
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|percent|%|USD|dollars?|euros?|pounds?))\b'
        entities["numbers"] = list(set(re.findall(number_pattern, text, re.IGNORECASE)))
        
        # Extract dates
        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\w+\s\d{1,2},?\s\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},?\s\d{4})\b'
        entities["dates"] = list(set(re.findall(date_pattern, text)))
        
        # Extract people names (simple pattern)
        people_pattern = r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
        potential_people = re.findall(people_pattern, text)
        # Filter out common false positives
        false_positives = ["New York", "Los Angeles", "United States", "North America", "South America"]
        entities["people"] = [name for name in set(potential_people) if name not in false_positives]
        
        total_entities = sum(len(v) for v in entities.values())
        logger.info(f"Extracted {total_entities} entities total")
        
        return {
            "entities": entities,
            "total_entities": total_entities,
            "source": "regex_extraction",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {
            "entities": {"organizations": [], "locations": [], "numbers": [], "dates": [], "people": []},
            "error": str(e),
            "source": "error",
            "success": False
        }

@mcp.tool()
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using simple keyword-based approach.
    
    Args:
        text: Text to analyze sentiment
    
    Returns:
        Dictionary with sentiment analysis results
    """
    try:
        logger.info(f"Analyzing sentiment of text (length: {len(text)})")
        
        if len(text) < 5:
            raise ValueError("Text too short for sentiment analysis")
        
        # Simple keyword-based sentiment analysis
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
            "positive", "success", "growth", "increase", "profit", "win", 
            "improve", "better", "best", "love", "like", "happy", "pleased"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "negative", "failure", 
            "decline", "decrease", "loss", "lose", "worst", "hate", "dislike", 
            "angry", "disappointed", "problem", "issue", "crisis", "risk"
        ]
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        neutral_count = len(words) - positive_count - negative_count
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.5
        elif positive_count > negative_count:
            sentiment = "positive"
            confidence = round(positive_count / total_sentiment_words, 2)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = round(negative_count / total_sentiment_words, 2)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "neutral_words": neutral_count,
            "total_words": len(words),
            "source": "keyword_analysis",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            "sentiment": "unknown",
            "error": str(e),
            "source": "error",
            "success": False
        }

@mcp.tool()
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "NLP Summarizer Server",
        "timestamp": __import__('time').time(),
        "azure_openai_available": llm_client and llm_client.client is not None,
        "available_operations": 4
    }

@mcp.tool()
def get_tools() -> List[Dict[str, Any]]:
    """Get list of available tools"""
    return [
        {
            "name": "summarize",
            "description": "Summarize text content using AI or local processing",
            "parameters": {
                "text": "string (required, min 10 chars) - Text to summarize",
                "max_length": "integer (optional, 50-2000) - Maximum summary length",
                "style": "string (optional) - Style: brief, detailed, bullet_points"
            }
        },
        {
            "name": "extract_entities",
            "description": "Extract named entities from text",
            "parameters": {
                "text": "string (required, min 10 chars) - Text to extract entities from"
            }
        },
        {
            "name": "analyze_sentiment",
            "description": "Analyze sentiment of text",
            "parameters": {
                "text": "string (required, min 5 chars) - Text to analyze"
            }
        },
        {
            "name": "health",
            "description": "Health check for the server",
            "parameters": {}
        }
    ]

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Starting NLP Summarizer MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9103")
    print(f"⚡ Server Name: NLP Summarizer Server")
    print(f"🤖 Azure OpenAI Available: {llm_client and llm_client.client is not None}")
    print(f"🔧 Available Operations: 4")
    print("=" * 60)
    
    # Run the server
    mcp.run(transport="http", host="0.0.0.0", port=9103)