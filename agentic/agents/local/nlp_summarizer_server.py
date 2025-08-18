# agentic/agents/local/nlp_summarizer_server.py
import json
import time
import uuid
import re
import sys
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agentic.core.llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server Implementation
class NLPSummarizerMCPServer:
    def __init__(self):
        self.app = FastAPI(title="NLP Summarizer MCP Server")
        self.sessions = {}
        self.tools = [
            {
                "name": "summarize",
                "description": "Summarize text content using Azure OpenAI with fallback to local processing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize", "minLength": 10},
                        "max_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 200, "description": "Maximum summary length in characters"},
                        "style": {"type": "string", "enum": ["brief", "detailed", "bullet_points"], "default": "brief", "description": "Summary style and format"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "extract_entities",
                "description": "Extract named entities from text (organizations, locations, numbers, dates)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to extract entities from", "minLength": 10}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "analyze_sentiment",
                "description": "Analyze sentiment of text using keyword-based approach",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze sentiment", "minLength": 5}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "health",
                "description": "Health check for the server",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        ]
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/")
        async def mcp_endpoint_post(
            request: Request,
            mcp_protocol_version: Optional[str] = Header(None, alias="MCP-Protocol-Version"),
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            origin: Optional[str] = Header(None)
        ):
            # Security: Validate Origin header
            if origin and not self._is_allowed_origin(origin):
                raise HTTPException(status_code=403, detail="Origin not allowed")
            
            try:
                body = await request.body()
                message = json.loads(body.decode('utf-8'))
                
                if message.get("method") == "initialize":
                    return await self._handle_initialize(message, mcp_protocol_version)
                elif message.get("method") == "initialized":
                    return JSONResponse(status_code=202, content={})
                elif message.get("method") == "tools/list":
                    return await self._handle_list_tools(message, mcp_session_id)
                elif message.get("method") == "tools/call":
                    return await self._handle_call_tool(message, mcp_session_id)
                else:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {"code": -32601, "message": "Method not found"}
                    }
                    return JSONResponse(content=error_response, status_code=400)
                    
            except json.JSONDecodeError:
                return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.get("/")
        async def mcp_endpoint_get(
            request: Request,
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            accept: Optional[str] = Header(None)
        ):
            if accept and "text/event-stream" in accept:
                return StreamingResponse(
                    self._sse_stream(mcp_session_id),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                return JSONResponse(content={"error": "SSE not supported"}, status_code=405)
    
    def _is_allowed_origin(self, origin: str) -> bool:
        allowed_origins = ["http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"]
        return any(origin.startswith(allowed) for allowed in allowed_origins)
    
    async def _handle_initialize(self, message: Dict[str, Any], protocol_version: Optional[str]) -> JSONResponse:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "protocol_version": protocol_version or "2025-06-18"
        }
        
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}, "logging": {}},
                "serverInfo": {"name": "NLP Summarizer MCP Server", "version": "1.0.0"}
            }
        }
        return JSONResponse(content=response, headers={"Mcp-Session-Id": session_id})
    
    async def _handle_list_tools(self, message: Dict[str, Any], session_id: Optional[str]) -> JSONResponse:
        if session_id and session_id not in self.sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=404)
        
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {"tools": self.tools}
        }
        return JSONResponse(content=response)
    
    async def _handle_call_tool(self, message: Dict[str, Any], session_id: Optional[str]) -> JSONResponse:
        if session_id and session_id not in self.sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=404)
        
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "summarize":
                result = await self._execute_summarize(arguments)
            elif tool_name == "extract_entities":
                result = await self._execute_extract_entities(arguments)
            elif tool_name == "analyze_sentiment":
                result = await self._execute_analyze_sentiment(arguments)
            elif tool_name == "health":
                result = await self._execute_health(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False)
                        }
                    ]
                }
            }
            return JSONResponse(content=response)
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32000, "message": str(e)}
            }
            return JSONResponse(content=error_response, status_code=500)
    
    async def _execute_summarize(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize text content using Azure OpenAI with fallback to local processing"""
        try:
            text = arguments.get("text", "")
            max_length = arguments.get("max_length", 200)
            style = arguments.get("style", "brief")
            
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
            return self._local_summarize(text, max_length, style)

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._local_summarize(text, max_length, style)
    
    def _local_summarize(self, text: str, max_length: int, style: str) -> Dict[str, Any]:
        """Local text summarization fallback"""
        try:
            cleaned = re.sub(r'\s+', ' ', text.strip())
            sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]
            key_points = self._extract_key_points(sentences)

            if style == "bullet_points":
                summary = "\n".join(f"• {pt}" for pt in key_points[:5])
            elif style == "detailed":
                important = self._select_important_sentences(sentences, max(3, len(sentences) // 3))
                summary = ". ".join(important)
            else:  # brief
                important = self._select_important_sentences(sentences, min(3, len(sentences)))
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
    
    def _extract_key_points(self, sentences: list) -> list:
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
    
    def _select_important_sentences(self, sentences: list, num_sentences: int) -> list:
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
    
    async def _execute_extract_entities(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities from text"""
        try:
            text = arguments.get("text", "")
            
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
            
            # Extract organizations
            org_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Corp|Inc|Ltd|LLC|Company|Industries|Group|Holdings|Corporation)\b'
            entities["organizations"] = list(set(re.findall(org_pattern, text)))
            
            # Extract locations
            location_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?:\s(?:City|State|Country|County|Province|Street|Avenue|Road))\b'
            entities["locations"] = list(set(re.findall(location_pattern, text)))
            
            # Extract numbers with units
            number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|percent|%|USD|dollars?|euros?|pounds?))\b'
            entities["numbers"] = list(set(re.findall(number_pattern, text, re.IGNORECASE)))
            
            # Extract dates
            date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\w+\s\d{1,2},?\s\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},?\s\d{4})\b'
            entities["dates"] = list(set(re.findall(date_pattern, text)))
            
            # Extract people names
            people_pattern = r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
            potential_people = re.findall(people_pattern, text)
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
    
    async def _execute_analyze_sentiment(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text using simple keyword-based approach"""
        try:
            text = arguments.get("text", "")
            
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
    
    async def _execute_health(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health check tool"""
        return {
            "status": "healthy",
            "server": "NLP Summarizer Server",
            "timestamp": time.time(),
            "azure_openai_available": llm_client and llm_client.client is not None,
            "available_operations": 4
        }
    
    async def _sse_stream(self, session_id: Optional[str]):
        """Generate SSE stream for server-to-client communication"""
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': time.time()})}\n\n"
        
        try:
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# Create server instance
server = NLPSummarizerMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Starting NLP Summarizer MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9103")
    print(f"⚡ Server Name: NLP Summarizer Server")
    print(f"🤖 Azure OpenAI Available: {llm_client and llm_client.client is not None}")
    print(f"🔧 Available Operations: 4")
    print(f"📋 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9103, log_level="info")