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
    
    def _get_session(self, agent_name: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for agent"""
        if agent_name not in self.sessions:
            self.sessions[agent_name] = aiohttp.ClientSession(timeout=self.timeout)
        return self.sessions[agent_name]
    
    async def list_tools(self, agent: AgentSpec) -> List[ToolSpec]:
        """List tools available on MCP server"""
        try:
            session = self._get_session(agent.name)
            async with session.post(
                f"{agent.endpoint}/tools/list",
                json={"method": "tools/list", "params": {}}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    tools = []
                    for tool_data in data.get("result", {}).get("tools", []):
                        tool = ToolSpec(
                            name=tool_data["name"],
                            description=tool_data.get("description", ""),
                            params_schema=tool_data.get("inputSchema", {}),
                            returns_schema=tool_data.get("outputSchema", {})
                        )
                        tools.append(tool)
                    return tools
        except Exception as e:
            logger.error(f"Failed to list tools for {agent.name}: {e}")
        
        # Fallback to agent definition
        return agent.tools
    
    async def call_tool(
        self, 
        agent: AgentSpec, 
        tool_name: str, 
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call a tool on MCP server"""
        try:
            session = self._get_session(agent.name)
            request_data = {
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }
            
            timeout_obj = aiohttp.ClientTimeout(total=timeout or settings.mcp_default_timeout_sec)
            
            async with session.post(
                f"{agent.endpoint}/tools/call",
                json=request_data,
                timeout=timeout_obj
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        return {
                            "success": True,
                            "data": data["result"].get("content", [{}])[0].get("text", ""),
                            "raw_response": data["result"]
                        }
                    else:
                        return {
                            "success": False,
                            "error": data.get("error", "Unknown error"),
                            "raw_response": data
                        }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Tool call timed out after {timeout or settings.mcp_default_timeout_sec} seconds"
            }
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {agent.name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

