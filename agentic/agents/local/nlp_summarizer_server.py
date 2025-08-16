# agentic/agents/local/nlp_summarizer_server.py
from fastmcp import FastMCP
from typing import Dict, Any
import re

mcp = FastMCP("NLP Summarizer Server")

@mcp.tool()
def summarize(text: str, max_length: int = 200, style: str = "brief") -> Dict[str, Any]:
    """Summarize text content using simple NLP techniques"""
    try:
        # Clean and prepare text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract key information
        key_points = _extract_key_points(sentences)
        
        # Generate summary based on style
        if style == "bullet_points":
            summary = "\n".join([f"• {point}" for point in key_points[:5]])
        elif style == "detailed":
            # Take more sentences for detailed summary
            important_sentences = _select_important_sentences(sentences, max(3, len(sentences)//3))
            summary = ". ".join(important_sentences)
        else:  # brief
            important_sentences = _select_important_sentences(sentences, min(3, len(sentences)))
            summary = ". ".join(important_sentences)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return {
            "summary": summary,
            "key_points": key_points,
            "word_count": len(summary.split()),
            "original_length": len(text),
            "compression_ratio": round(len(summary) / len(text), 2) if text else 0
        }
        
    except Exception as e:
        return {
            "summary": "Error generating summary",
            "key_points": [],
            "error": str(e),
            "word_count": 0
        }

def _extract_key_points(sentences: list) -> list:
    """Extract key points from sentences"""
    key_points = []
    
    # Simple keyword-based extraction
    important_keywords = [
        "market", "trend", "growth", "increase", "decrease", "analysis", 
        "research", "study", "report", "data", "statistics", "percent",
        "million", "billion", "company", "industry", "consumer", "customer"
    ]
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in important_keywords):
            if len(sentence) > 20 and len(sentence) < 200:  # Good length for key point
                key_points.append(sentence.strip())
        
        if len(key_points) >= 10:  # Limit key points
            break
    
    return key_points[:8]  # Return top 8 key points

def _select_important_sentences(sentences: list, num_sentences: int) -> list:
    """Select most important sentences for summary"""
    if len(sentences) <= num_sentences:
        return sentences
    
    # Score sentences based on various factors
    scored_sentences = []
    
    for sentence in sentences:
        score = 0
        words = sentence.lower().split()
        
        # Length score (prefer medium length)
        if 10 <= len(words) <= 25:
            score += 2
        elif 5 <= len(words) < 10 or 25 < len(words) <= 35:
            score += 1
        
        # Keyword score
        important_words = ["market", "research", "analysis", "trend", "data", "study", "report"]
        score += sum(1 for word in words if word in important_words)
        
        # Position score (prefer earlier sentences)
        position_bonus = max(0, 2 - (sentences.index(sentence) / len(sentences)) * 2)
        score += position_bonus
        
        scored_sentences.append((sentence, score))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    selected = [sent[0] for sent in scored_sentences[:num_sentences]]
    
    # Maintain original order
    result = []
    for sentence in sentences:
        if sentence in selected:
            result.append(sentence)
        if len(result) >= num_sentences:
            break
    
    return result

if __name__ == "__main__":
    print("Starting NLP Summarizer Server on port 9103...")
    # mcp.run(port=9103)
    mcp.run(transport="sse", host="0.0.0.0", port=9103)
