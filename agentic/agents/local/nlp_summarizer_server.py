# agentic/agents/local/nlp_summarizer_server.py
import asyncio
import re
from typing import Dict, Any
from fastmcp import FastMCP

mcp = FastMCP("NLP Summarizer Server")

@mcp.tool()
def summarize(text: str, max_length: int = 200, style: str = "brief") -> Dict[str, Any]:
    """
    Summarize text content using simple NLP techniques, supporting
    'brief', 'detailed', and 'bullet_points' styles.
    """
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
    keywords = [
        "market", "trend", "growth", "increase", "decrease", "analysis", 
        "research", "study", "report", "data", "statistics", "percent",
        "million", "billion", "company", "industry", "consumer", "customer"
    ]
    points = []
    for s in sentences:
        if any(k in s.lower() for k in keywords) and 20 < len(s) < 200:
            points.append(s)
        if len(points) >= 8:
            break
    return points

def _select_important_sentences(sentences: list, num_sentences: int) -> list:
    if len(sentences) <= num_sentences:
        return sentences

    scored = []
    for idx, s in enumerate(sentences):
        words = s.lower().split()
        score = 0
        score += 2 if 10 <= len(words) <= 25 else 1 if 5 <= len(words) < 10 or 25 < len(words) <= 35 else 0
        score += sum(1 for w in words if w in ["market", "research", "analysis", "trend", "data", "study", "report"])
        pos_bonus = max(0, 2 - (idx / len(sentences)) * 2)
        score += pos_bonus
        scored.append((s, score))

    selected = [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:num_sentences]]
    result = [s for s in sentences if s in selected]
    return result[:num_sentences]

async def main():
    print("Starting NLP Summarizer Server on port 9103 (async mode)...")
    await mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=9103,
        log_level="info"
    )

if __name__ == "__main__":
    asyncio.run(main())
