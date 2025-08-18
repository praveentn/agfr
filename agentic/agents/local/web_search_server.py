# agentic/agents/local/web_search_server.py
import json
import time
import uuid
import sys
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentic.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import search engine if available
try:
    from agentic.modules.search_engine import WebSearchEngine
    search_engine = WebSearchEngine()
    SEARCH_AVAILABLE = True
    logger.info("Real search engine loaded successfully")
except ImportError:
    logger.warning("Search engine module not available, using fallback")
    search_engine = None
    SEARCH_AVAILABLE = False

# MCP Server Implementation
class WebSearchMCPServer:
    def __init__(self):
        self.app = FastAPI(title="Web Search MCP Server")
        self.sessions = {}
        self.tools = [
            {
                "name": "search",
                "description": "Search the web for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8, "description": "Maximum number of results"},
                        "recency_days": {"type": "integer", "minimum": 0, "description": "Only return results from last N days"}
                    },
                    "required": ["query"]
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
            # Security: Validate Origin header for DNS rebinding protection
            if origin and not self._is_allowed_origin(origin):
                raise HTTPException(status_code=403, detail="Origin not allowed")
            
            try:
                body = await request.body()
                message = json.loads(body.decode('utf-8'))
                
                # Handle JSON-RPC message
                if message.get("method") == "initialize":
                    return await self._handle_initialize(message, mcp_protocol_version)
                elif message.get("method") == "initialized":
                    return JSONResponse(status_code=202, content={})
                elif message.get("method") == "tools/list":
                    return await self._handle_list_tools(message, mcp_session_id)
                elif message.get("method") == "tools/call":
                    return await self._handle_call_tool(message, mcp_session_id)
                else:
                    # Unknown method
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": -32601,
                            "message": "Method not found"
                        }
                    }
                    return JSONResponse(content=error_response, status_code=400)
                    
            except json.JSONDecodeError:
                return JSONResponse(
                    content={"error": "Invalid JSON"}, 
                    status_code=400
                )
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return JSONResponse(
                    content={"error": str(e)}, 
                    status_code=500
                )
        
        @self.app.get("/")
        async def mcp_endpoint_get(
            request: Request,
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            accept: Optional[str] = Header(None)
        ):
            # Check if client accepts SSE
            if accept and "text/event-stream" in accept:
                return StreamingResponse(
                    self._sse_stream(mcp_session_id),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            else:
                return JSONResponse(
                    content={"error": "SSE not supported"}, 
                    status_code=405
                )
    
    def _is_allowed_origin(self, origin: str) -> bool:
        """Validate origin to prevent DNS rebinding attacks"""
        allowed_origins = [
            "http://localhost",
            "http://127.0.0.1",
            "https://localhost",
            "https://127.0.0.1"
        ]
        return any(origin.startswith(allowed) for allowed in allowed_origins)
    
    async def _handle_initialize(self, message: Dict[str, Any], protocol_version: Optional[str]) -> JSONResponse:
        """Handle MCP initialize request"""
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
                "capabilities": {
                    "tools": {},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "Web Search MCP Server",
                    "version": "1.0.0"
                }
            }
        }
        
        return JSONResponse(
            content=response,
            headers={"Mcp-Session-Id": session_id}
        )
    
    async def _handle_list_tools(self, message: Dict[str, Any], session_id: Optional[str]) -> JSONResponse:
        """Handle tools/list request"""
        if session_id and session_id not in self.sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=404)
        
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "tools": self.tools
            }
        }
        return JSONResponse(content=response)
    
    async def _handle_call_tool(self, message: Dict[str, Any], session_id: Optional[str]) -> JSONResponse:
        """Handle tools/call request"""
        if session_id and session_id not in self.sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=404)
        
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "search":
                result = await self._execute_search(arguments)
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
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
            return JSONResponse(content=error_response, status_code=500)
    
    async def _execute_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search tool"""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 8)
        recency_days = arguments.get("recency_days")
        
        logger.info(f"Processing search request: query='{query}', limit={limit}")
        
        if SEARCH_AVAILABLE and search_engine:
            # Use real search engine
            search_type = "general"
            if any(keyword in query.lower() for keyword in ["market", "industry", "competitive", "analysis"]):
                search_type = "company"
            elif any(keyword in query.lower() for keyword in ["startup", "funding", "venture"]):
                search_type = "startup"
            elif any(keyword in query.lower() for keyword in ["technology", "tech", "innovation"]):
                search_type = "technology"
            
            logger.info(f"Performing real search for: {query} (type: {search_type})")
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
                "source": "real_search",
                "success": True
            }
        else:
            # Fallback to enhanced mock results
            return self._get_enhanced_mock_results(query, limit)
    
    async def _execute_health(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health check tool"""
        return {
            "status": "healthy",
            "server": "Web Search Server",
            "timestamp": time.time(),
            "search_engine_available": SEARCH_AVAILABLE
        }
    
    def _get_enhanced_mock_results(self, query: str, limit: int) -> Dict[str, Any]:
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
                }
            ]
        elif "france" in query.lower() and "capital" in query.lower():
            mock_results = [
                {
                    "title": "Paris - Capital and Largest City of France",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "snippet": "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km².",
                    "date_published": "2024-01-01",
                    "domain": "wikipedia.org",
                    "relevance_score": 0.98
                }
            ]
        else:
            # Generic results based on query
            topic = query.replace("market research", "").replace("search for", "").strip()
            mock_results = [
                {
                    "title": f"Comprehensive Guide to {topic.title()}",
                    "url": f"https://www.comprehensive-guides.com/{topic.lower().replace(' ', '-')}",
                    "snippet": f"Complete overview and analysis of {topic}, including latest developments, key insights, and expert perspectives on current trends.",
                    "date_published": "2024-01-15",
                    "domain": "comprehensive-guides.com",
                    "relevance_score": 0.7
                }
            ]
        
        return {
            "items": mock_results[:limit],
            "total_found": len(mock_results),
            "query": query,
            "timestamp": time.time(),
            "source": "mock_search",
            "success": True
        }
    
    async def _sse_stream(self, session_id: Optional[str]):
        """Generate SSE stream for server-to-client communication"""
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': time.time()})}\n\n"
        
        # Keep connection alive
        try:
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# Create server instance
server = WebSearchMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Starting Web Search MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9101")
    print(f"🔧 Search Engine Available: {SEARCH_AVAILABLE}")
    print(f"⚡ Server Name: Web Search Server")
    print(f"📋 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=9101, log_level="info")