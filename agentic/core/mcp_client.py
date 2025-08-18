# agentic/core/mcp_client.py
import asyncio
import aiohttp
import json
import uuid
import time
from typing import Dict, Any, List, Optional, Callable
import logging
from .types import ToolSpec, AgentSpec
from .config import settings

logger = logging.getLogger(__name__)

class MCPClientManager:
    def __init__(self):
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.agent_sessions: Dict[str, str] = {}  # agent_name -> session_id
        self.timeout = aiohttp.ClientTimeout(total=settings.mcp_default_timeout_sec)
        self.max_retries = 3
        self.retry_delay = 1.0
        self._agent_health_cache = {}
        self._tools_cache = {}
        self.protocol_version = "2025-06-18"
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
    
    async def close_all(self):
        for session in self.sessions.values():
            if not session.closed:
                await session.close()
        self.sessions.clear()
        self.agent_sessions.clear()
        self._agent_health_cache.clear()
        self._tools_cache.clear()
    
    def _get_session(self, agent_name: str) -> aiohttp.ClientSession:
        if agent_name not in self.sessions:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.sessions[agent_name] = aiohttp.ClientSession(
                timeout=self.timeout, 
                connector=connector,
                headers={
                    "MCP-Protocol-Version": self.protocol_version,
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
        return self.sessions[agent_name]

    async def _initialize_agent_session(self, agent: AgentSpec) -> bool:
        """Initialize MCP session with agent"""
        try:
            session = self._get_session(agent.name)
            
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {
                        "roots": {
                            "listChanged": False
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "Agentic Framework",
                        "version": "1.0.0"
                    }
                }
            }
            
            async with session.post(agent.endpoint, json=init_request) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("result"):
                        # Extract session ID from headers
                        session_id = response.headers.get("Mcp-Session-Id")
                        if session_id:
                            self.agent_sessions[agent.name] = session_id
                            logger.info(f"Initialized MCP session for {agent.name}: {session_id}")
                        
                        # Send initialized notification
                        init_notification = {
                            "jsonrpc": "2.0",
                            "method": "initialized",
                            "params": {}
                        }
                        
                        headers = {}
                        if session_id:
                            headers["Mcp-Session-Id"] = session_id
                        
                        async with session.post(agent.endpoint, json=init_notification, headers=headers) as init_response:
                            if init_response.status == 202:
                                logger.info(f"Agent {agent.name} initialization complete")
                                return True
                
                logger.error(f"Failed to initialize agent {agent.name}: {response.status}")
                return False
                
        except Exception as e:
            logger.error(f"Agent initialization failed for {agent.name}: {e}")
            return False

    async def call_tool(self, agent: AgentSpec, tool_name: str, params: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Call tool using MCP protocol"""
        logger.info(f"Calling MCP tool {tool_name} on agent {agent.name}")
        
        # Ensure agent session is initialized
        if agent.name not in self.agent_sessions:
            if not await self._initialize_agent_session(agent):
                return {
                    "success": False, 
                    "error": f"Failed to initialize session with agent {agent.name}"
                }
        
        for attempt in range(self.max_retries):
            try:
                session = self._get_session(agent.name)
                session_id = self.agent_sessions.get(agent.name)
                
                # Prepare MCP tool call request
                tool_call_request = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": params
                    }
                }
                
                headers = {}
                if session_id:
                    headers["Mcp-Session-Id"] = session_id
                
                timeout_obj = aiohttp.ClientTimeout(total=timeout or 30)
                
                async with session.post(
                    agent.endpoint, 
                    json=tool_call_request, 
                    headers=headers,
                    timeout=timeout_obj
                ) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            return self._process_mcp_response(data)
                        except:
                            text_data = await response.text()
                            return {"success": True, "data": text_data}
                    elif response.status == 404:
                        # Session expired, reinitialize
                        logger.warning(f"Session expired for {agent.name}, reinitializing...")
                        if agent.name in self.agent_sessions:
                            del self.agent_sessions[agent.name]
                        
                        if await self._initialize_agent_session(agent):
                            continue  # Retry with new session
                        else:
                            return {"success": False, "error": "Failed to reinitialize session"}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
                        
            except Exception as e:
                logger.error(f"MCP tool call failed for {agent.name}:{tool_name}, attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        return {"success": False, "error": f"Failed to call tool {tool_name} on {agent.name} after {self.max_retries} attempts"}

    async def test_agent_connection(self, agent: AgentSpec) -> bool:
        """Test if agent is reachable and healthy using MCP protocol"""
        try:
            # Check cache first (valid for 30 seconds)
            cache_key = f"{agent.name}_health"
            cached_result = self._agent_health_cache.get(cache_key)
            if cached_result and (time.time() - cached_result['timestamp']) < 30:
                return cached_result['healthy']
            
            # Try to initialize session first
            if agent.name not in self.agent_sessions:
                if not await self._initialize_agent_session(agent):
                    self._agent_health_cache[cache_key] = {
                        'healthy': False,
                        'timestamp': time.time()
                    }
                    return False
            
            # Try calling health tool
            try:
                health_result = await self.call_tool(agent, "health", {}, timeout=5)
                healthy = health_result.get("success", False)
                self._agent_health_cache[cache_key] = {
                    'healthy': healthy,
                    'timestamp': time.time()
                }
                return healthy
            except Exception as e:
                logger.debug(f"Health tool check failed for {agent.name}: {e}")
                
                # Fallback to basic connectivity test
                session = self._get_session(agent.name)
                session_id = self.agent_sessions.get(agent.name)
                
                headers = {}
                if session_id:
                    headers["Mcp-Session-Id"] = session_id
                
                async with session.get(
                    agent.endpoint,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    healthy = response.status < 500
                    self._agent_health_cache[cache_key] = {
                        'healthy': healthy,
                        'timestamp': time.time()
                    }
                    return healthy
                
        except Exception as e:
            logger.debug(f"Health check failed for {agent.name}: {e}")
            self._agent_health_cache[cache_key] = {
                'healthy': False,
                'timestamp': time.time()
            }
            return False
    
    async def get_agent_tools(self, agent: AgentSpec) -> List[Dict[str, Any]]:
        """Get list of tools available on an agent using MCP protocol"""
        try:
            # Check cache first
            cache_key = f"{agent.name}_tools"
            cached_result = self._tools_cache.get(cache_key)
            if cached_result and (time.time() - cached_result['timestamp']) < 300:  # 5 min cache
                return cached_result['tools']
            
            # Ensure agent session is initialized
            if agent.name not in self.agent_sessions:
                if not await self._initialize_agent_session(agent):
                    return []
            
            session = self._get_session(agent.name)
            session_id = self.agent_sessions.get(agent.name)
            
            # Prepare MCP tools/list request
            tools_list_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/list",
                "params": {}
            }
            
            headers = {}
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            
            async with session.post(
                agent.endpoint, 
                json=tools_list_request, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("result") and data["result"].get("tools"):
                        tools = data["result"]["tools"]
                        self._tools_cache[cache_key] = {
                            'tools': tools,
                            'timestamp': time.time()
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

    async def subscribe_stream(self, agent: AgentSpec, callback: Callable[[str], None]):
        """
        Subscribe to SSE stream of an MCP server using GET endpoint.
        `callback` will be called with each event data.
        """
        try:
            # Ensure agent session is initialized
            if agent.name not in self.agent_sessions:
                if not await self._initialize_agent_session(agent):
                    logger.error(f"Failed to initialize session for SSE subscription to {agent.name}")
                    return
            
            session = self._get_session(agent.name)
            session_id = self.agent_sessions.get(agent.name)
            
            headers = {
                "Accept": "text/event-stream"
            }
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            
            async with session.get(agent.endpoint, headers=headers) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        if line:
                            text = line.decode("utf-8").strip()
                            if text.startswith("data:"):
                                await callback(text[5:].strip())
                else:
                    logger.error(f"SSE subscription failed for {agent.name}: HTTP {resp.status}")
                    
        except Exception as e:
            logger.error(f"Stream subscription failed for {agent.name}: {e}")

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
                    "timestamp": time.time(),
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
                        "timestamp": time.time(),
                        "source": "local_fallback"
                    }
                }
            
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
