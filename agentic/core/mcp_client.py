# agentic/core/mcp_client.py
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
import logging
from .types import ToolSpec, AgentSpec
from .config import settings

logger = logging.getLogger(__name__)

class MCPClientManager:
    def __init__(self):
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.timeout = aiohttp.ClientTimeout(total=settings.mcp_default_timeout_sec)
        self.max_retries = 2  # Reduced retries for faster failure
        self.retry_delay = 0.5
        self.sse_connections: Dict[str, Any] = {}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
    
    async def close_all(self):
        """Close all HTTP sessions and SSE connections"""
        for session in self.sessions.values():
            if not session.closed:
                await session.close()
        self.sessions.clear()
        self.sse_connections.clear()
    
    def _get_session(self, agent_name: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for agent"""
        if agent_name not in self.sessions:
            connector = aiohttp.TCPConnector(
                limit=10, 
                limit_per_host=5,
                timeout=aiohttp.ClientTimeout(connect=5, total=30)  # Shorter timeouts
            )
            self.sessions[agent_name] = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector
            )
        return self.sessions[agent_name]
    
    async def test_agent_connection(self, agent: AgentSpec) -> bool:
        """Test if agent is reachable"""
        try:
            session = self._get_session(agent.name)
            timeout_obj = aiohttp.ClientTimeout(total=5)  # Quick connection test
            
            async with session.get(
                f"{agent.endpoint}/health",
                timeout=timeout_obj
            ) as response:
                return response.status == 200
        except:
            # Try alternate health check endpoint
            try:
                async with session.get(
                    f"{agent.endpoint}/",
                    timeout=timeout_obj
                ) as response:
                    return response.status in [200, 404]  # 404 is acceptable for basic health
            except:
                return False
    
    async def call_tool(
        self, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call a tool on MCP server with improved error handling"""
        logger.info(f"Calling tool {tool_name} on agent {agent.name} with params: {params}")
        
        # Test connection first
        if not await self.test_agent_connection(agent):
            logger.warning(f"Agent {agent.name} appears unreachable, trying direct call anyway")
        
        for attempt in range(self.max_retries):
            try:
                session = self._get_session(agent.name)
                timeout_obj = aiohttp.ClientTimeout(total=timeout or 15)  # Shorter default timeout
                
                # Try direct tool endpoint first
                try:
                    result = await self._call_direct_endpoint(session, agent, tool_name, params, timeout_obj)
                    if result["success"]:
                        logger.info(f"Tool {tool_name} succeeded on {agent.name}")
                        return result
                except Exception as e:
                    logger.debug(f"Direct endpoint failed: {e}")
                
                # Try HTTP POST to root with tool call
                try:
                    result = await self._call_http_post(session, agent, tool_name, params, timeout_obj)
                    if result["success"]:
                        logger.info(f"Tool {tool_name} succeeded on {agent.name} via HTTP POST")
                        return result
                except Exception as e:
                    logger.debug(f"HTTP POST failed: {e}")
                
                # If agent server is local, try direct function call as last resort
                if "localhost" in agent.endpoint or "127.0.0.1" in agent.endpoint:
                    try:
                        result = await self._call_local_fallback(agent, tool_name, params)
                        if result["success"]:
                            logger.info(f"Tool {tool_name} succeeded on {agent.name} via local fallback")
                            return result
                    except Exception as e:
                        logger.debug(f"Local fallback failed: {e}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Tool call timed out for {agent.name}:{tool_name}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Tool call failed for {agent.name}:{tool_name}, attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
        
        # All attempts failed
        logger.error(f"All attempts failed for {agent.name}:{tool_name}")
        return {
            "success": False,
            "error": f"Failed to call tool {tool_name} on {agent.name} after {self.max_retries} attempts"
        }
    
    async def _call_direct_endpoint(
        self, 
        session: aiohttp.ClientSession, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: aiohttp.ClientTimeout
    ) -> Dict[str, Any]:
        """Call tool via direct endpoint"""
        url = f"{agent.endpoint}/tools/{tool_name}"
        
        async with session.post(url, json=params, timeout=timeout) as response:
            if response.status == 200:
                data = await response.json()
                return {"success": True, "data": data}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def _call_http_post(
        self, 
        session: aiohttp.ClientSession, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: aiohttp.ClientTimeout
    ) -> Dict[str, Any]:
        """Call tool via HTTP POST to root endpoint"""
        request_data = {
            "tool": tool_name,
            "params": params
        }
        
        async with session.post(f"{agent.endpoint}", json=request_data, timeout=timeout) as response:
            if response.status == 200:
                data = await response.json()
                return {"success": True, "data": data}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def _call_local_fallback(
        self, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback for local agents - direct function simulation"""
        try:
            # Simulate responses based on tool type
            if tool_name == "search":
                # Import and use web search directly
                try:
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents', 'local'))
                    from web_search_server import search
                    
                    result = search(
                        query=params.get('query', ''),
                        limit=params.get('limit', 8),
                        recency_days=params.get('recency_days')
                    )
                    return {"success": True, "data": result}
                except Exception:
                    pass
            
            elif tool_name == "summarize":
                # Use Azure OpenAI client directly
                from .llm_client import llm_client
                if llm_client:
                    result = await llm_client.summarize_text(
                        text=params.get('text', ''),
                        style=params.get('style', 'brief'),
                        max_length=params.get('max_length', 200)
                    )
                    return {"success": True, "data": result}
            
            elif tool_name == "tabulate":
                # Simple tabulation
                data = params.get('data', [])
                fields = params.get('fields', [])
                format_type = params.get('format', 'json')
                
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        data = [{"content": data}]
                
                if not isinstance(data, list):
                    data = [data]
                
                result = {
                    "table": data,
                    "format": format_type,
                    "row_count": len(data),
                    "columns": fields or (list(data[0].keys()) if data and isinstance(data[0], dict) else [])
                }
                return {"success": True, "data": result}
            
            elif tool_name in ["add", "multiply", "divide", "subtract", "power"]:
                # Calculator operations
                a = params.get('a', 0)
                b = params.get('b', 1)
                
                try:
                    if tool_name == "add":
                        result = a + b
                    elif tool_name == "multiply":
                        result = a * b
                    elif tool_name == "divide":
                        result = a / b if b != 0 else "Error: Division by zero"
                    elif tool_name == "subtract":
                        result = a - b
                    else:  # power
                        result = a ** b
                    
                    return {"success": True, "data": result}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            return {"success": False, "error": f"Local fallback failed: {str(e)}"}
    
    def _process_mcp_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP response format"""
        if "result" in data:
            result_data = data["result"]
            if "content" in result_data and isinstance(result_data["content"], list):
                # Extract text from content array
                content_text = result_data["content"][0].get("text", "")
                
                # Try to parse as JSON, otherwise return as string
                try:
                    parsed_content = json.loads(content_text)
                    return {
                        "success": True,
                        "data": parsed_content,
                        "raw_response": data["result"]
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "data": content_text,
                        "raw_response": data["result"]
                    }
            else:
                # Direct result without content wrapper
                return {
                    "success": True,
                    "data": result_data,
                    "raw_response": data["result"]
                }
        elif "error" in data:
            return {
                "success": False,
                "error": data["error"].get("message", "Unknown error"),
                "raw_response": data
            }
        else:
            return {
                "success": False,
                "error": "Invalid response format",
                "raw_response": data
            }
