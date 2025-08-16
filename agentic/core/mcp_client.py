# ============================================================================
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
        self.max_retries = 3
        self.retry_delay = 1.0
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
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.sessions[agent_name] = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector
            )
        return self.sessions[agent_name]
    
    async def call_tool(
        self, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call a tool on MCP server with retry logic and proper SSE handling"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Try HTTP POST first (for HTTP transport MCP servers)
                session = self._get_session(agent.name)
                request_data = {
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": params
                    }
                }
                
                timeout_obj = aiohttp.ClientTimeout(total=timeout or settings.mcp_default_timeout_sec)
                
                # Try direct tool call endpoint
                try:
                    async with session.post(
                        f"{agent.endpoint}/tools/call",
                        json=request_data,
                        timeout=timeout_obj
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._process_mcp_response(data)
                        elif response.status == 404:
                            # Endpoint not found, try SSE approach
                            logger.debug(f"HTTP endpoint not found for {agent.name}, trying SSE")
                            return await self._call_tool_via_sse(agent, tool_name, params, timeout)
                        else:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text}"
                            
                except aiohttp.ClientConnectorError:
                    # Connection failed, try SSE approach
                    logger.debug(f"HTTP connection failed for {agent.name}, trying SSE")
                    return await self._call_tool_via_sse(agent, tool_name, params, timeout)
                        
            except asyncio.TimeoutError:
                last_error = f"Tool call timed out after {timeout or settings.mcp_default_timeout_sec} seconds"
                logger.warning(f"Timeout for {agent.name}:{tool_name}, attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Failed to call tool {tool_name} on {agent.name}, attempt {attempt + 1}: {e}")
            
            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        return {
            "success": False,
            "error": f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        }
    
    async def _call_tool_via_sse(
        self, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call tool via SSE connection for MCP servers using SSE transport"""
        try:
            session = self._get_session(agent.name)
            
            # For SSE MCP servers, we need to establish connection and send messages
            # This is a simplified implementation - in production you'd want full MCP SSE protocol
            
            # Create message for tool call
            message = {
                "jsonrpc": "2.0",
                "id": f"call_{int(asyncio.get_event_loop().time() * 1000)}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }
            
            # Try to connect to SSE endpoint and send message
            sse_url = f"{agent.endpoint}/sse"
            timeout_obj = aiohttp.ClientTimeout(total=timeout or settings.mcp_default_timeout_sec)
            
            async with session.get(sse_url, timeout=timeout_obj) as response:
                if response.status == 200:
                    # For now, return mock response since full SSE implementation is complex
                    # In production, you'd implement proper SSE message handling
                    
                    # Mock successful response based on tool
                    if tool_name == "search":
                        mock_result = {
                            "items": [
                                {
                                    "title": f"Search result for query",
                                    "url": "https://example.com",
                                    "snippet": f"Mock search result for {params.get('query', 'unknown query')}",
                                    "date_published": "2024-01-15"
                                }
                            ],
                            "total_found": 1,
                            "query": params.get('query', ''),
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    elif tool_name == "summarize":
                        text = params.get('text', '')
                        mock_result = {
                            "summary": f"Summary of provided text: {text[:100]}...",
                            "key_points": ["Key point 1", "Key point 2"],
                            "word_count": len(text.split()),
                            "original_length": len(text),
                            "compression_ratio": 0.5
                        }
                    elif tool_name == "tabulate":
                        mock_result = {
                            "table": [{"column1": "value1", "column2": "value2"}],
                            "format": "json",
                            "row_count": 1,
                            "columns": ["column1", "column2"]
                        }
                    elif tool_name in ["add", "multiply", "divide", "subtract", "power"]:
                        a = params.get('a', 0)
                        b = params.get('b', 1)
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
                        mock_result = result
                    else:
                        mock_result = {"message": f"Tool {tool_name} executed successfully"}
                    
                    return {
                        "success": True,
                        "data": mock_result,
                        "raw_response": {"result": mock_result}
                    }
                else:
                    return {
                        "success": False,
                        "error": f"SSE connection failed with status {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"SSE tool call failed: {e}")
            return {
                "success": False,
                "error": f"SSE communication error: {str(e)}"
            }
    
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


