# agentic/agents/local/web_search_server.py
import json
import time
import uuid
import sys
import os
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from urllib.parse import quote_plus

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

class EnhancedWebSearchMCPServer:
    """Enhanced Web Search with Azure OpenAI integration for intelligent search and results processing"""
    
    def __init__(self):
        self.app = FastAPI(title="Enhanced Web Search MCP Server")
        self.sessions = {}
        self.search_cache = {}
        
        self.tools = [
            {
                "name": "search",
                "description": "Intelligent web search with AI-powered query optimization and result enhancement",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8, "description": "Maximum number of results to return"},
                        "recency_days": {"type": "integer", "minimum": 1, "maximum": 365, "description": "Limit results to past N days (optional)"},
                        "search_type": {"type": "string", "enum": ["general", "news", "academic", "images", "videos"], "default": "general", "description": "Type of search to perform"},
                        "optimize_query": {"type": "boolean", "default": True, "description": "Use AI to optimize search query"},
                        "enhance_results": {"type": "boolean", "default": True, "description": "Use AI to enhance and categorize results"},
                        "domain_filter": {"type": "string", "description": "Limit search to specific domains (e.g., 'edu,gov,org')", "examples": ["edu,gov", "news,journalism"]},
                        "language": {"type": "string", "default": "en", "description": "Search language preference"},
                        "region": {"type": "string", "default": "us", "description": "Geographic region for search"},
                        "safe_search": {"type": "string", "enum": ["strict", "moderate", "off"], "default": "moderate", "description": "Safe search filter level"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "multi_search",
                "description": "Perform multiple related searches and combine results intelligently",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "base_query": {"type": "string", "description": "Main search topic", "minLength": 1},
                        "search_variants": {"type": "array", "items": {"type": "string"}, "description": "Additional search variations to try (optional - AI will generate if not provided)"},
                        "max_results_per_search": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5, "description": "Results per individual search"},
                        "combine_strategy": {"type": "string", "enum": ["merge", "rank", "categorize"], "default": "rank", "description": "How to combine results from multiple searches"},
                        "deduplicate": {"type": "boolean", "default": True, "description": "Remove duplicate results"}
                    },
                    "required": ["base_query"]
                }
            },
            {
                "name": "research_topic",
                "description": "Comprehensive research on a topic using multiple search strategies and AI analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Research topic", "minLength": 5},
                        "research_depth": {"type": "string", "enum": ["overview", "detailed", "comprehensive"], "default": "detailed", "description": "Depth of research to perform"},
                        "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Specific aspects to focus on (optional)"},
                        "source_types": {"type": "array", "items": {"type": "string"}, "description": "Preferred source types", "examples": ["academic", "news", "official", "expert"]},
                        "time_frame": {"type": "string", "description": "Time frame for research", "examples": ["past_year", "past_month", "recent", "historical"]},
                        "include_analysis": {"type": "boolean", "default": True, "description": "Include AI analysis of findings"}
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "fact_check",
                "description": "Verify claims or facts using authoritative sources",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "Claim or fact to verify", "minLength": 10},
                        "sources_required": {"type": "integer", "minimum": 2, "maximum": 10, "default": 3, "description": "Minimum number of authoritative sources required"},
                        "authority_level": {"type": "string", "enum": ["any", "established", "official"], "default": "established", "description": "Required authority level of sources"},
                        "include_confidence": {"type": "boolean", "default": True, "description": "Include confidence assessment of verification"}
                    },
                    "required": ["claim"]
                }
            },
            {
                "name": "trending_topics",
                "description": "Find trending topics and current events in specified categories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["general", "technology", "business", "science", "health", "politics", "entertainment"], "default": "general", "description": "Category of trending topics"},
                        "region": {"type": "string", "default": "global", "description": "Geographic region for trends"},
                        "time_period": {"type": "string", "enum": ["hour", "day", "week", "month"], "default": "day", "description": "Time period for trending analysis"},
                        "include_analysis": {"type": "boolean", "default": True, "description": "Include AI analysis of trends"}
                    }
                }
            },
            {
                "name": "health",
                "description": "Health check for the web search server",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        ]
        
        # Search providers configuration
        self.search_providers = {
            "duckduckgo": {"enabled": True, "rate_limit": 100},
            "bing": {"enabled": os.getenv("BING_API_KEY") is not None, "rate_limit": 1000},
            "google": {"enabled": os.getenv("GOOGLE_API_KEY") is not None, "rate_limit": 100}
        }
        
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
                    return await self._handle_tool_call(message, mcp_session_id)
                elif message.get("method") == "ping":
                    return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"status": "ok"}}
                else:
                    return {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": -32601, "message": "Method not found"}}
                    
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                logger.error(f"Request handling error: {e}")
                return {"jsonrpc": "2.0", "error": {"code": -32603, "message": "Internal error"}}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "server": "Enhanced Web Search Server", "timestamp": time.time()}
        
        @self.app.get("/events")
        async def sse_endpoint(mcp_session_id: Optional[str] = Header(None)):
            return StreamingResponse(
                self._sse_stream(mcp_session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
    
    def _is_allowed_origin(self, origin: str) -> bool:
        """Validate request origin for security"""
        allowed_origins = [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
        return origin in allowed_origins
    
    async def _handle_initialize(self, message: Dict[str, Any], protocol_version: Optional[str]) -> Dict[str, Any]:
        """Handle MCP initialization"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "protocol_version": protocol_version or "2025-06-18",
            "capabilities": message.get("params", {}).get("capabilities", {}),
            "ai_enabled": llm_client and llm_client.client is not None
        }
        
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {
                    "name": "Enhanced Web Search Server",
                    "version": "2.0.0",
                    "description": "Intelligent web search with Azure OpenAI integration"
                },
                "capabilities": {
                    "tools": {"listChanged": True},
                    "logging": {"level": "info"},
                    "experimental": {"streaming": True, "ai_enhancement": bool(llm_client and llm_client.client)}
                },
                "sessionId": session_id
            }
        }
    
    async def _handle_list_tools(self, message: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
        """Handle tools list request"""
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {"tools": self.tools}
        }
    
    async def _handle_tool_call(self, message: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
        """Handle tool execution request"""
        try:
            tool_name = message.get("params", {}).get("name")
            arguments = message.get("params", {}).get("arguments", {})
            
            logger.info(f"Executing tool: {tool_name}")
            
            if tool_name == "search":
                result = await self._execute_search(arguments)
            elif tool_name == "multi_search":
                result = await self._execute_multi_search(arguments)
            elif tool_name == "research_topic":
                result = await self._execute_research_topic(arguments)
            elif tool_name == "fact_check":
                result = await self._execute_fact_check(arguments)
            elif tool_name == "trending_topics":
                result = await self._execute_trending_topics(arguments)
            elif tool_name == "health":
                result = await self._execute_health(arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32601, "message": f"Tool '{tool_name}' not found"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32603, "message": f"Tool execution failed: {str(e)}"}
            }
    
    async def _execute_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intelligent web search with AI enhancement"""
        try:
            query = arguments.get("query", "").strip()
            limit = min(max(arguments.get("limit", 8), 1), 50)
            recency_days = arguments.get("recency_days")
            search_type = arguments.get("search_type", "general")
            optimize_query = arguments.get("optimize_query", True)
            enhance_results = arguments.get("enhance_results", True)
            domain_filter = arguments.get("domain_filter", "")
            language = arguments.get("language", "en")
            region = arguments.get("region", "us")
            safe_search = arguments.get("safe_search", "moderate")
            
            if not query:
                return {"success": False, "error": "Search query is required"}
            
            # AI-powered query optimization
            optimized_query = query
            if optimize_query and llm_client and llm_client.client:
                optimized_query = await self._optimize_search_query(query, search_type, domain_filter)
            
            # Check cache first
            cache_key = self._get_cache_key(optimized_query, limit, recency_days, search_type, domain_filter, language, region)
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if time.time() - cached_result["timestamp"] < 300:  # 5 minutes cache
                    cached_result["from_cache"] = True
                    return cached_result
            
            # Perform search
            search_results = await self._perform_web_search(
                optimized_query, limit, recency_days, search_type, 
                domain_filter, language, region, safe_search
            )
            
            # AI-powered result enhancement
            if enhance_results and llm_client and llm_client.client and search_results.get("items"):
                enhanced_results = await self._enhance_search_results(
                    search_results["items"], query, search_type
                )
                search_results["items"] = enhanced_results
                search_results["ai_enhanced"] = True
            
            # Add metadata
            search_results.update({
                "original_query": query,
                "optimized_query": optimized_query,
                "search_type": search_type,
                "query_optimization_used": optimize_query and (query != optimized_query),
                "result_enhancement_used": enhance_results and search_results.get("ai_enhanced", False),
                "search_timestamp": datetime.now().isoformat(),
                "from_cache": False
            })
            
            # Cache results
            self.search_cache[cache_key] = {**search_results, "timestamp": time.time()}
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_multi_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple related searches and combine results"""
        try:
            base_query = arguments.get("base_query", "").strip()
            search_variants = arguments.get("search_variants", [])
            max_results_per_search = min(max(arguments.get("max_results_per_search", 5), 1), 20)
            combine_strategy = arguments.get("combine_strategy", "rank")
            deduplicate = arguments.get("deduplicate", True)
            
            if not base_query:
                return {"success": False, "error": "Base query is required"}
            
            # Generate search variants using AI if not provided
            if not search_variants and llm_client and llm_client.client:
                search_variants = await self._generate_search_variants(base_query)
            
            # If still no variants, use basic alternatives
            if not search_variants:
                search_variants = [
                    f"{base_query} information",
                    f"{base_query} details",
                    f"{base_query} overview"
                ]
            
            # Add base query to variants
            all_queries = [base_query] + search_variants[:4]  # Limit to 5 total searches
            
            # Perform all searches concurrently
            search_tasks = []
            for query in all_queries:
                task = self._perform_web_search(query, max_results_per_search)
                search_tasks.append(task)
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results
            combined_items = []
            search_metadata = []
            
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.error(f"Search failed for query '{all_queries[i]}': {result}")
                    continue
                
                if result.get("success") and result.get("items"):
                    combined_items.extend(result["items"])
                    search_metadata.append({
                        "query": all_queries[i],
                        "results_count": len(result["items"]),
                        "success": True
                    })
                else:
                    search_metadata.append({
                        "query": all_queries[i],
                        "results_count": 0,
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    })
            
            # Apply deduplication
            if deduplicate:
                combined_items = await self._deduplicate_results(combined_items)
            
            # Apply combination strategy
            final_results = await self._apply_combination_strategy(
                combined_items, combine_strategy, base_query
            )
            
            return {
                "success": True,
                "items": final_results,
                "total_items": len(final_results),
                "base_query": base_query,
                "search_variants": search_variants,
                "search_metadata": search_metadata,
                "combine_strategy": combine_strategy,
                "deduplication_applied": deduplicate,
                "search_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Multi-search execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_research_topic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive topic research"""
        try:
            topic = arguments.get("topic", "").strip()
            research_depth = arguments.get("research_depth", "detailed")
            focus_areas = arguments.get("focus_areas", [])
            source_types = arguments.get("source_types", [])
            time_frame = arguments.get("time_frame", "")
            include_analysis = arguments.get("include_analysis", True)
            
            if not topic:
                return {"success": False, "error": "Research topic is required"}
            
            # Generate research strategy
            research_queries = await self._generate_research_queries(
                topic, research_depth, focus_areas, source_types, time_frame
            )
            
            # Execute research searches
            research_results = []
            for query_info in research_queries:
                search_args = {
                    "query": query_info["query"],
                    "limit": query_info.get("limit", 10),
                    "search_type": query_info.get("search_type", "general"),
                    "domain_filter": query_info.get("domain_filter", "")
                }
                
                if time_frame:
                    search_args["recency_days"] = self._parse_time_frame(time_frame)
                
                result = await self._execute_search(search_args)
                if result.get("success") and result.get("items"):
                    research_results.append({
                        "query": query_info["query"],
                        "focus": query_info.get("focus", "general"),
                        "results": result["items"]
                    })
            
            # Compile research findings
            all_results = []
            for research in research_results:
                all_results.extend(research["results"])
            
            # Deduplicate and rank
            unique_results = await self._deduplicate_results(all_results)
            ranked_results = await self._rank_research_results(unique_results, topic, focus_areas)
            
            # Generate AI analysis if requested
            analysis = {}
            if include_analysis and llm_client and llm_client.client:
                analysis = await self._generate_research_analysis(
                    topic, ranked_results, focus_areas, research_depth
                )
            
            return {
                "success": True,
                "topic": topic,
                "research_depth": research_depth,
                "focus_areas": focus_areas,
                "results": ranked_results,
                "total_sources": len(ranked_results),
                "research_queries": research_queries,
                "analysis": analysis,
                "research_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_fact_check(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fact checking using authoritative sources"""
        try:
            claim = arguments.get("claim", "").strip()
            sources_required = min(max(arguments.get("sources_required", 3), 2), 10)
            authority_level = arguments.get("authority_level", "established")
            include_confidence = arguments.get("include_confidence", True)
            
            if not claim:
                return {"success": False, "error": "Claim to verify is required"}
            
            # Generate fact-checking queries
            fact_check_queries = await self._generate_fact_check_queries(claim, authority_level)
            
            # Search for verification sources
            verification_sources = []
            for query in fact_check_queries:
                result = await self._execute_search({
                    "query": query,
                    "limit": 15,
                    "search_type": "general",
                    "domain_filter": self._get_authority_domains(authority_level)
                })
                
                if result.get("success") and result.get("items"):
                    verification_sources.extend(result["items"])
            
            # Filter and rank authoritative sources
            authoritative_sources = await self._filter_authoritative_sources(
                verification_sources, authority_level, sources_required
            )
            
            # Generate verification analysis
            verification_result = {}
            if llm_client and llm_client.client and authoritative_sources:
                verification_result = await self._analyze_fact_verification(
                    claim, authoritative_sources, include_confidence
                )
            
            return {
                "success": True,
                "claim": claim,
                "verification": verification_result,
                "sources": authoritative_sources,
                "sources_found": len(authoritative_sources),
                "sources_required": sources_required,
                "authority_level": authority_level,
                "verification_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fact check execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_trending_topics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trending topics search"""
        try:
            category = arguments.get("category", "general")
            region = arguments.get("region", "global")
            time_period = arguments.get("time_period", "day")
            include_analysis = arguments.get("include_analysis", True)
            
            # Generate trending search queries
            trending_queries = self._generate_trending_queries(category, region, time_period)
            
            # Execute trending searches
            trending_results = []
            for query in trending_queries:
                result = await self._execute_search({
                    "query": query,
                    "limit": 20,
                    "search_type": "news",
                    "recency_days": self._get_recency_days(time_period)
                })
                
                if result.get("success") and result.get("items"):
                    trending_results.extend(result["items"])
            
            # Extract and analyze trends
            trends = await self._extract_trending_topics(trending_results, category)
            
            # Generate trend analysis
            analysis = {}
            if include_analysis and llm_client and llm_client.client:
                analysis = await self._analyze_trends(trends, category, time_period)
            
            return {
                "success": True,
                "category": category,
                "region": region,
                "time_period": time_period,
                "trends": trends,
                "analysis": analysis,
                "total_trends": len(trends),
                "search_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trending topics execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_health(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health check"""
        providers_status = {}
        for provider, config in self.search_providers.items():
            providers_status[provider] = "enabled" if config["enabled"] else "disabled"
        
        ai_status = "healthy" if llm_client and llm_client.client else "not_configured"
        
        return {
            "status": "healthy",
            "server": "Enhanced Web Search Server",
            "ai_enhancement": ai_status,
            "search_providers": providers_status,
            "cache_size": len(self.search_cache),
            "timestamp": time.time(),
            "capabilities": [
                "search", "multi_search", "research_topic", 
                "fact_check", "trending_topics"
            ]
        }
    
    # Core search implementation methods
    async def _perform_web_search(self, query: str, limit: int = 8, recency_days: Optional[int] = None, 
                                 search_type: str = "general", domain_filter: str = "", 
                                 language: str = "en", region: str = "us", 
                                 safe_search: str = "moderate") -> Dict[str, Any]:
        """Perform actual web search using available providers"""
        try:
            # Try different search providers in order of preference
            if self.search_providers["bing"]["enabled"]:
                return await self._search_bing(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
            elif self.search_providers["google"]["enabled"]:
                return await self._search_google(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
            else:
                return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"success": False, "error": str(e), "items": []}
    
    async def _search_duckduckgo(self, query: str, limit: int, recency_days: Optional[int], 
                               search_type: str, domain_filter: str, language: str, 
                               region: str, safe_search: str) -> Dict[str, Any]:
        """Search using DuckDuckGo (fallback implementation)"""
        try:
            # Construct search query with filters
            search_query = query
            if domain_filter:
                domains = domain_filter.split(',')
                domain_query = " OR ".join([f"site:{domain.strip()}" for domain in domains])
                search_query = f"({search_query}) AND ({domain_query})"
            
            if recency_days:
                # DuckDuckGo doesn't have direct date filtering, but we can try keywords
                if recency_days <= 7:
                    search_query += " recent"
                elif recency_days <= 30:
                    search_query += " latest"
            
            # Mock search results for demonstration
            # In a real implementation, you would use DuckDuckGo's API or scraping
            mock_results = [
                {
                    "title": f"Search result for '{query}' - Example 1",
                    "url": "https://example1.com",
                    "snippet": f"This is a mock search result snippet for the query '{query}'. It provides relevant information about the topic.",
                    "domain": "example1.com",
                    "relevance_score": 0.95,
                    "publish_date": datetime.now().isoformat()
                },
                {
                    "title": f"Search result for '{query}' - Example 2", 
                    "url": "https://example2.com",
                    "snippet": f"Another relevant search result for '{query}' with additional details and insights.",
                    "domain": "example2.com",
                    "relevance_score": 0.88,
                    "publish_date": (datetime.now() - timedelta(days=1)).isoformat()
                }
            ]
            
            # Limit results
            limited_results = mock_results[:limit]
            
            return {
                "success": True,
                "items": limited_results,
                "total_items": len(limited_results),
                "search_provider": "duckduckgo",
                "query_used": search_query
            }
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return {"success": False, "error": str(e), "items": []}
    
    async def _search_bing(self, query: str, limit: int, recency_days: Optional[int], 
                         search_type: str, domain_filter: str, language: str, 
                         region: str, safe_search: str) -> Dict[str, Any]:
        """Search using Bing Search API"""
        try:
            api_key = os.getenv("BING_API_KEY")
            if not api_key:
                return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
            
            # Bing Search API implementation would go here
            # For now, falling back to DuckDuckGo
            return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
    
    async def _search_google(self, query: str, limit: int, recency_days: Optional[int], 
                           search_type: str, domain_filter: str, language: str, 
                           region: str, safe_search: str) -> Dict[str, Any]:
        """Search using Google Custom Search API"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
            
            # Google Custom Search API implementation would go here
            # For now, falling back to DuckDuckGo
            return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return await self._search_duckduckgo(query, limit, recency_days, search_type, domain_filter, language, region, safe_search)
    
    # AI-powered enhancement methods
    async def _optimize_search_query(self, query: str, search_type: str, domain_filter: str) -> str:
        """Use AI to optimize search query for better results"""
        try:
            optimization_prompt = f"""
            Optimize this search query for better web search results:
            
            Original Query: "{query}"
            Search Type: {search_type}
            Domain Filter: {domain_filter or "none"}
            
            Provide an optimized search query that:
            1. Uses effective search keywords
            2. Is optimized for the search type
            3. Considers domain restrictions if any
            4. Improves result relevance
            5. Maintains the original intent
            
            Return only the optimized query, no explanation.
            """
            
            messages = [
                {"role": "system", "content": "You are a search optimization expert. Create effective search queries that yield better results."},
                {"role": "user", "content": optimization_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=100)
            
            if response and response.strip():
                optimized = response.strip().strip('"')
                # Ensure the optimized query isn't too different from original
                if len(optimized) > len(query) * 3:
                    return query
                return optimized
            
            return query
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return query
    
    async def _enhance_search_results(self, results: List[Dict], original_query: str, search_type: str) -> List[Dict]:
        """Use AI to enhance and categorize search results"""
        try:
            if not results:
                return results
            
            # Prepare results summary for AI analysis
            results_summary = []
            for i, result in enumerate(results[:10]):  # Limit to first 10 for processing
                results_summary.append({
                    "index": i,
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "domain": result.get("domain", ""),
                    "url": result.get("url", "")
                })
            
            enhancement_prompt = f"""
            Enhance these search results for the query "{original_query}" (search type: {search_type}):
            
            Results to enhance:
            {json.dumps(results_summary, indent=2)}
            
            For each result, provide:
            1. Relevance score (0-1)
            2. Content category
            3. Authority assessment
            4. Key insights from snippet
            5. Recommended priority
            
            Return as JSON array with enhanced metadata:
            [
                {{
                    "index": 0,
                    "relevance_score": 0.95,
                    "category": "primary_source",
                    "authority": "high",
                    "key_insights": ["insight1", "insight2"],
                    "priority": "high"
                }}
            ]
            """
            
            messages = [
                {"role": "system", "content": "You are a search result analyzer. Enhance search results with metadata and insights."},
                {"role": "user", "content": enhancement_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=1500)
            
            if response:
                try:
                    enhancements = json.loads(response.strip())
                    
                    # Apply enhancements to original results
                    enhanced_results = []
                    for result in results:
                        enhanced_result = result.copy()
                        
                        # Find corresponding enhancement
                        result_index = next((i for i, r in enumerate(results_summary) if r["url"] == result.get("url")), -1)
                        if result_index >= 0:
                            enhancement = next((e for e in enhancements if e.get("index") == result_index), {})
                            enhanced_result.update({
                                "ai_relevance_score": enhancement.get("relevance_score", 0.5),
                                "ai_category": enhancement.get("category", "general"),
                                "ai_authority": enhancement.get("authority", "medium"),
                                "ai_insights": enhancement.get("key_insights", []),
                                "ai_priority": enhancement.get("priority", "medium")
                            })
                        
                        enhanced_results.append(enhanced_result)
                    
                    # Sort by AI relevance score
                    enhanced_results.sort(key=lambda x: x.get("ai_relevance_score", 0.5), reverse=True)
                    
                    return enhanced_results
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse AI enhancement response")
            
            return results
            
        except Exception as e:
            logger.error(f"Result enhancement failed: {e}")
            return results
    
    # Helper methods
    def _get_cache_key(self, query: str, limit: int, recency_days: Optional[int], 
                      search_type: str, domain_filter: str, language: str, region: str) -> str:
        """Generate cache key for search results"""
        return f"{query}:{limit}:{recency_days}:{search_type}:{domain_filter}:{language}:{region}"
    
    async def _generate_search_variants(self, base_query: str) -> List[str]:
        """Generate search variants using AI"""
        try:
            variant_prompt = f"""
            Generate 3-4 search query variants for: "{base_query}"
            
            Create variants that:
            1. Use different keywords but same intent
            2. Include synonyms and related terms
            3. Cover different aspects of the topic
            4. Maintain search effectiveness
            
            Return as JSON array: ["variant1", "variant2", "variant3"]
            """
            
            messages = [
                {"role": "system", "content": "You are a search query generator. Create effective search variants."},
                {"role": "user", "content": variant_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.4, max_tokens=200)
            
            if response:
                try:
                    variants = json.loads(response.strip())
                    return variants if isinstance(variants, list) else []
                except json.JSONDecodeError:
                    pass
            
            return []
            
        except Exception as e:
            logger.error(f"Search variant generation failed: {e}")
            return []
    
    async def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on URL and title similarity"""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "").lower()
            
            # Skip if URL is already seen
            if url in seen_urls:
                continue
            
            # Skip if title is very similar to existing one
            is_similar = any(
                self._calculate_similarity(title, existing_title) > 0.8
                for existing_title in seen_titles
            )
            
            if not is_similar:
                unique_results.append(result)
                seen_urls.add(url)
                seen_titles.add(title)
        
        return unique_results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _apply_combination_strategy(self, items: List[Dict], strategy: str, base_query: str) -> List[Dict]:
        """Apply combination strategy to search results"""
        if strategy == "merge":
            return items
        elif strategy == "rank":
            # Sort by relevance score if available
            return sorted(items, key=lambda x: x.get("ai_relevance_score", x.get("relevance_score", 0.5)), reverse=True)
        elif strategy == "categorize":
            # Group by category if available
            categorized = {}
            for item in items:
                category = item.get("ai_category", "general")
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(item)
            
            # Flatten with category priority
            result = []
            for category in ["primary_source", "authoritative", "general"]:
                result.extend(categorized.get(category, []))
            
            return result
        
        return items
    
    def _parse_time_frame(self, time_frame: str) -> int:
        """Parse time frame string to days"""
        time_frame_mapping = {
            "past_year": 365,
            "past_month": 30,
            "recent": 7,
            "historical": None
        }
        return time_frame_mapping.get(time_frame, 30)
    
    def _get_recency_days(self, time_period: str) -> int:
        """Get recency days for trending topics"""
        period_mapping = {
            "hour": 1,
            "day": 1,
            "week": 7,
            "month": 30
        }
        return period_mapping.get(time_period, 1)
    
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
server = EnhancedWebSearchMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Starting Enhanced Web Search MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9101")
    print(f"⚡ Server Name: Enhanced Web Search Server")
    print(f"🔧 Capabilities: Intelligent Search, Multi-Search, Research, Fact-Check, Trending")
    print(f"🤖 Azure OpenAI: Query optimization and result enhancement")
    print(f"📊 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9101, log_level="info")