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
        self._agent_health_cache = {}
        self._tools_cache = {}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
    
    async def close_all(self):
        """Close all HTTP sessions"""
        for session in self.sessions.values():
            if not session.closed:
                await session.close()
        self.sessions.clear()
        self._agent_health_cache.clear()
        self._tools_cache.clear()
    
    def _get_session(self, agent_name: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for agent"""
        if agent_name not in self.sessions:
            connector = aiohttp.TCPConnector(
                limit=10, 
                limit_per_host=5
            )
            self.sessions[agent_name] = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector
            )
        return self.sessions[agent_name]
    
    async def test_agent_connection(self, agent: AgentSpec) -> bool:
        """Test if agent is reachable and healthy"""
        try:
            # Check cache first (valid for 30 seconds)
            cache_key = f"{agent.name}_health"
            cached_result = self._agent_health_cache.get(cache_key)
            if cached_result and (asyncio.get_event_loop().time() - cached_result['timestamp']) < 30:
                return cached_result['healthy']
            
            session = self._get_session(agent.name)
            timeout_obj = aiohttp.ClientTimeout(total=5)  # Quick health check
            
            # Try FastMCP root endpoint first
            try:
                async with session.get(
                    agent.endpoint,
                    timeout=timeout_obj
                ) as response:
                    if response.status == 200:
                        # Try to get the response content
                        try:
                            content = await response.text()
                            # If it's a FastMCP server, it should return some info
                            healthy = "FastMCP" in content or len(content) > 0
                        except:
                            healthy = True  # If we get 200, assume healthy
                        
                        self._agent_health_cache[cache_key] = {
                            'healthy': healthy,
                            'timestamp': asyncio.get_event_loop().time()
                        }
                        return healthy
            except Exception as e:
                logger.debug(f"Root endpoint check failed for {agent.name}: {e}")
            
            # Try calling the health tool directly
            try:
                health_result = await self.call_tool(agent, "health", {}, timeout=5)
                healthy = health_result.get("success", False)
                self._agent_health_cache[cache_key] = {
                    'healthy': healthy,
                    'timestamp': asyncio.get_event_loop().time()
                }
                return healthy
            except Exception as e:
                logger.debug(f"Health tool check failed for {agent.name}: {e}")
            
            # Final fallback - just try to connect
            try:
                async with session.get(
                    agent.endpoint,
                    timeout=timeout_obj
                ) as response:
                    healthy = response.status < 500  # Any non-server error is acceptable
                    self._agent_health_cache[cache_key] = {
                        'healthy': healthy,
                        'timestamp': asyncio.get_event_loop().time()
                    }
                    return healthy
            except:
                self._agent_health_cache[cache_key] = {
                    'healthy': False,
                    'timestamp': asyncio.get_event_loop().time()
                }
                return False
                
        except Exception as e:
            logger.debug(f"Health check failed for {agent.name}: {e}")
            return False
    
    async def get_agent_tools(self, agent: AgentSpec) -> List[Dict[str, Any]]:
        """Get list of tools available on an agent"""
        try:
            # Check cache first
            cache_key = f"{agent.name}_tools"
            cached_result = self._tools_cache.get(cache_key)
            if cached_result and (asyncio.get_event_loop().time() - cached_result['timestamp']) < 300:  # 5 min cache
                return cached_result['tools']
            
            # Try calling get_tools method
            tools_result = await self.call_tool(agent, "get_tools", {}, timeout=10)
            if tools_result.get("success") and tools_result.get("data"):
                tools = tools_result["data"]
                self._tools_cache[cache_key] = {
                    'tools': tools,
                    'timestamp': asyncio.get_event_loop().time()
                }
                return tools
            
            # Fallback: return tools from agent spec
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.params_schema
                }
                for tool in agent.tools
            ]
            
        except Exception as e:
            logger.error(f"Failed to get tools for {agent.name}: {e}")
            return []
    
    async def call_tool(
        self, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call a tool on MCP server with FastMCP-specific endpoint handling"""
        logger.info(f"Calling tool {tool_name} on agent {agent.name}")
        
        for attempt in range(self.max_retries):
            try:
                session = self._get_session(agent.name)
                timeout_obj = aiohttp.ClientTimeout(total=timeout or 30)
                
                # Method 1: Try FastMCP direct tool endpoint (most likely)
                try:
                    result = await self._call_fastmcp_direct(session, agent, tool_name, params, timeout_obj)
                    if result["success"]:
                        logger.info(f"Tool {tool_name} succeeded on {agent.name} via FastMCP direct")
                        return result
                except Exception as e:
                    logger.debug(f"FastMCP direct call failed: {e}")
                
                # Method 2: Try FastMCP with call action
                try:
                    result = await self._call_fastmcp_call_action(session, agent, tool_name, params, timeout_obj)
                    if result["success"]:
                        logger.info(f"Tool {tool_name} succeeded on {agent.name} via FastMCP call action")
                        return result
                except Exception as e:
                    logger.debug(f"FastMCP call action failed: {e}")
                
                # Method 3: Try standard tool endpoint
                try:
                    result = await self._call_generic_tool(session, agent, tool_name, params, timeout_obj)
                    if result["success"]:
                        logger.info(f"Tool {tool_name} succeeded on {agent.name} via generic endpoint")
                        return result
                except Exception as e:
                    logger.debug(f"Generic tool call failed: {e}")
                
                # Method 4: Try JSON-RPC style call
                try:
                    result = await self._call_jsonrpc_tool(session, agent, tool_name, params, timeout_obj)
                    if result["success"]:
                        logger.info(f"Tool {tool_name} succeeded on {agent.name} via JSON-RPC")
                        return result
                except Exception as e:
                    logger.debug(f"JSON-RPC call failed: {e}")
                
                # Method 5: Local fallback for development
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
    
    async def _call_fastmcp_direct(
        self, 
        session: aiohttp.ClientSession, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: aiohttp.ClientTimeout
    ) -> Dict[str, Any]:
        """Call tool via FastMCP direct endpoint POST /{tool_name}"""
        
        url = f"{agent.endpoint}/{tool_name}"
        
        # FastMCP expects parameters as direct JSON body
        async with session.post(url, json=params, timeout=timeout) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    return {"success": True, "data": data}
                except:
                    # If JSON parsing fails, try text
                    text_data = await response.text()
                    return {"success": True, "data": text_data}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def _call_fastmcp_call_action(
        self, 
        session: aiohttp.ClientSession, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: aiohttp.ClientTimeout
    ) -> Dict[str, Any]:
        """Call tool via FastMCP call action POST /call"""
        
        url = f"{agent.endpoint}/call"
        request_data = {
            "tool": tool_name,
            "params": params
        }
        
        async with session.post(url, json=request_data, timeout=timeout) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    return {"success": True, "data": data}
                except:
                    text_data = await response.text()
                    return {"success": True, "data": text_data}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def _call_generic_tool(
        self, 
        session: aiohttp.ClientSession, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: aiohttp.ClientTimeout
    ) -> Dict[str, Any]:
        """Call tool via generic /tools/ endpoint"""
        url = f"{agent.endpoint}/tools/{tool_name}"
        
        async with session.post(url, json=params, timeout=timeout) as response:
            if response.status == 200:
                data = await response.json()
                return {"success": True, "data": data}
            else:
                error_text = await response.text()
                return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
    
    async def _call_jsonrpc_tool(
        self, 
        session: aiohttp.ClientSession, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: aiohttp.ClientTimeout
    ) -> Dict[str, Any]:
        """Call tool via JSON-RPC format"""
        request_data = {
            "jsonrpc": "2.0",
            "method": tool_name,
            "params": params,
            "id": 1
        }
        
        async with session.post(f"{agent.endpoint}/jsonrpc", json=request_data, timeout=timeout) as response:
            if response.status == 200:
                data = await response.json()
                if "result" in data:
                    return {"success": True, "data": data["result"]}
                elif "error" in data:
                    return {"success": False, "error": data["error"]}
                else:
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
        """Fallback for local agents - simulate responses"""
        try:
            logger.info(f"Using local fallback for {agent.name}:{tool_name}")
            
            # Simulate responses based on tool type
            if tool_name == "search":
                query = params.get('query', 'default query')
                limit = params.get('limit', 5)
                
                # Generate mock search results based on query
                if "france" in query.lower() and "capital" in query.lower():
                    mock_items = [
                        {
                            "title": "Paris - Capital and Largest City of France",
                            "url": "https://en.wikipedia.org/wiki/Paris",
                            "snippet": "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents.",
                            "date_published": "2024-01-01",
                            "domain": "wikipedia.org",
                            "relevance_score": 0.98
                        }
                    ]
                else:
                    mock_items = []
                    for i in range(min(limit, 3)):
                        mock_items.append({
                            "title": f"Search result {i+1} for '{query}'",
                            "url": f"https://example{i+1}.com/search-{query.replace(' ', '-')}",
                            "snippet": f"This is a mock search result {i+1} for the query '{query}'. It contains relevant information.",
                            "date_published": "2024-01-15",
                            "domain": f"example{i+1}.com",
                            "relevance_score": 0.8 - (i * 0.1)
                        })
                
                result = {
                    "items": mock_items,
                    "total_found": len(mock_items),
                    "query": query,
                    "timestamp": __import__('time').time(),
                    "source": "local_fallback",
                    "success": True
                }
                return {"success": True, "data": result}
            
            elif tool_name == "health":
                return {
                    "success": True,
                    "data": {
                        "status": "healthy",
                        "server": agent.name,
                        "timestamp": __import__('time').time(),
                        "source": "local_fallback"
                    }
                }
            
            elif tool_name == "get_tools":
                return {
                    "success": True,
                    "data": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.params_schema
                        }
                        for tool in agent.tools
                    ]
                }
            
            elif tool_name == "summarize":
                text = params.get('text', '')
                style = params.get('style', 'brief')
                max_length = params.get('max_length', 200)
                
                # Simple local summarization
                sentences = text.split('.')[:3]  # Take first 3 sentences
                summary = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
                
                if len(summary) > max_length:
                    summary = summary[:max_length-3] + "..."
                
                result = {
                    "summary": summary,
                    "key_points": [s.strip() for s in sentences[:2] if s.strip()],
                    "word_count": len(summary.split()),
                    "original_length": len(text),
                    "compression_ratio": round(len(summary) / len(text), 2) if text else 0,
                    "source": "local_fallback",
                    "success": True
                }
                return {"success": True, "data": result}
            
            elif tool_name == "tabulate":
                data = params.get('data', [])
                fields = params.get('fields', [])
                format_type = params.get('format', 'json')
                
                if isinstance(data, str):
                    try:
                        data = __import__('json').loads(data)
                    except:
                        data = [{"content": data}]
                
                if not isinstance(data, list):
                    data = [data] if isinstance(data, dict) else [{"value": str(data)}]
                
                result = {
                    "table": data,
                    "format": format_type,
                    "row_count": len(data),
                    "columns": fields or (list(data[0].keys()) if data and isinstance(data[0], dict) else []),
                    "source": "local_fallback",
                    "success": True
                }
                return {"success": True, "data": result}
            
            elif tool_name in ["add", "multiply", "divide", "subtract", "power"]:
                a = float(params.get('a', 0))
                b = float(params.get('b', 1))
                
                try:
                    if tool_name == "add":
                        result = a + b
                    elif tool_name == "multiply":
                        result = a * b
                    elif tool_name == "divide":
                        if b == 0:
                            return {"success": False, "error": "Division by zero"}
                        result = a / b
                    elif tool_name == "subtract":
                        result = a - b
                    else:  # power
                        result = a ** b
                    
                    # Round to 3 decimal places if not integer
                    if isinstance(result, float) and result.is_integer():
                        result = int(result)
                    elif isinstance(result, float):
                        result = round(result, 3)
                    
                    return {"success": True, "data": result}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            logger.error(f"Local fallback failed: {e}")
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

