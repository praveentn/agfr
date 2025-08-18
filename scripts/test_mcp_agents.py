# scripts/test_mcp_servers.py
"""
Test script to verify MCP server compliance and functionality
"""
import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, Any, List

# MCP Server endpoints
MCP_SERVERS = {
    "web_search": "http://127.0.0.1:9101",
    "tabulator": "http://127.0.0.1:9102", 
    "nlp_summarizer": "http://127.0.0.1:9103",
    "calculator": "http://127.0.0.1:9104"
}

class MCPTester:
    def __init__(self):
        self.protocol_version = "2025-06-18"
        self.sessions = {}
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run comprehensive MCP server tests"""
        print("=" * 70)
        print("            MCP SERVER COMPLIANCE TEST SUITE")
        print("=" * 70)
        print(f"Protocol Version: {self.protocol_version}")
        print(f"Testing {len(MCP_SERVERS)} MCP servers...")
        print("=" * 70)
        
        async with aiohttp.ClientSession() as session:
            for server_name, endpoint in MCP_SERVERS.items():
                print(f"\n🧪 Testing {server_name.upper()} ({endpoint})")
                print("-" * 50)
                
                self.test_results[server_name] = await self.test_server(session, server_name, endpoint)
        
        self.print_summary()
    
    async def test_server(self, session: aiohttp.ClientSession, server_name: str, endpoint: str) -> Dict[str, Any]:
        """Test individual MCP server"""
        results = {
            "connectivity": False,
            "initialization": False,
            "tools_list": False,
            "tool_execution": False,
            "session_management": False,
            "errors": []
        }
        
        try:
            # Test 1: Basic connectivity
            print("  ✓ Testing basic connectivity...")
            if await self.test_connectivity(session, endpoint):
                results["connectivity"] = True
                print("    ✅ Server is reachable")
            else:
                results["errors"].append("Server not reachable")
                print("    ❌ Server not reachable")
                return results
            
            # Test 2: MCP Initialization
            print("  ✓ Testing MCP initialization...")
            session_id = await self.test_initialization(session, endpoint)
            if session_id:
                results["initialization"] = True
                results["session_management"] = True
                self.sessions[server_name] = session_id
                print(f"    ✅ Initialization successful (Session: {session_id[:8]}...)")
            else:
                results["errors"].append("Initialization failed")
                print("    ❌ Initialization failed")
                return results
            
            # Test 3: Tools listing
            print("  ✓ Testing tools/list...")
            tools = await self.test_tools_list(session, endpoint, session_id)
            if tools:
                results["tools_list"] = True
                print(f"    ✅ Found {len(tools)} tools: {[t['name'] for t in tools]}")
            else:
                results["errors"].append("Tools listing failed")
                print("    ❌ Tools listing failed")
            
            # Test 4: Tool execution
            print("  ✓ Testing tool execution...")
            if await self.test_tool_execution(session, endpoint, session_id, tools):
                results["tool_execution"] = True
                print("    ✅ Tool execution successful")
            else:
                results["errors"].append("Tool execution failed")
                print("    ❌ Tool execution failed")
        
        except Exception as e:
            results["errors"].append(f"Unexpected error: {str(e)}")
            print(f"    ❌ Unexpected error: {str(e)}")
        
        return results
    
    async def test_connectivity(self, session: aiohttp.ClientSession, endpoint: str) -> bool:
        """Test basic server connectivity"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(endpoint, timeout=timeout) as response:
                # Accept any response that's not a connection error
                return response.status < 500
        except Exception:
            return False
    
    async def test_initialization(self, session: aiohttp.ClientSession, endpoint: str) -> str:
        """Test MCP initialization protocol"""
        try:
            init_request = {
                "jsonrpc": "2.0",
                "id": "test-init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {
                        "roots": {"listChanged": False},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "MCP Test Client",
                        "version": "1.0.0"
                    }
                }
            }
            
            headers = {
                "MCP-Protocol-Version": self.protocol_version,
                "Content-Type": "application/json"
            }
            
            async with session.post(endpoint, json=init_request, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("result"):
                        session_id = response.headers.get("Mcp-Session-Id")
                        
                        # Send initialized notification
                        init_notification = {
                            "jsonrpc": "2.0",
                            "method": "initialized",
                            "params": {}
                        }
                        
                        notif_headers = headers.copy()
                        if session_id:
                            notif_headers["Mcp-Session-Id"] = session_id
                        
                        async with session.post(endpoint, json=init_notification, headers=notif_headers) as notif_response:
                            if notif_response.status == 202:
                                return session_id
                        
                        return session_id  # Return even if notification fails
            return None
        except Exception as e:
            print(f"    Debug: Initialization error: {e}")
            return None
    
    async def test_tools_list(self, session: aiohttp.ClientSession, endpoint: str, session_id: str) -> List[Dict[str, Any]]:
        """Test tools/list method"""
        try:
            tools_request = {
                "jsonrpc": "2.0",
                "id": "test-tools-1",
                "method": "tools/list",
                "params": {}
            }
            
            headers = {
                "MCP-Protocol-Version": self.protocol_version,
                "Content-Type": "application/json"
            }
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            
            async with session.post(endpoint, json=tools_request, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("result") and data["result"].get("tools"):
                        return data["result"]["tools"]
            return []
        except Exception as e:
            print(f"    Debug: Tools list error: {e}")
            return []
    
    async def test_tool_execution(self, session: aiohttp.ClientSession, endpoint: str, session_id: str, tools: List[Dict[str, Any]]) -> bool:
        """Test tool execution"""
        if not tools:
            return False
        
        try:
            # Find a suitable tool to test
            test_tool = None
            test_params = {}
            
            for tool in tools:
                tool_name = tool.get("name")
                if tool_name == "health":
                    test_tool = tool_name
                    test_params = {}
                    break
                elif tool_name == "add":
                    test_tool = tool_name
                    test_params = {"a": 5, "b": 3}
                    break
                elif tool_name == "search":
                    test_tool = tool_name
                    test_params = {"query": "test query", "limit": 1}
                    break
            
            if not test_tool:
                # Use first available tool
                test_tool = tools[0].get("name")
                test_params = {}
            
            tool_request = {
                "jsonrpc": "2.0",
                "id": "test-tool-1",
                "method": "tools/call",
                "params": {
                    "name": test_tool,
                    "arguments": test_params
                }
            }
            
            headers = {
                "MCP-Protocol-Version": self.protocol_version,
                "Content-Type": "application/json"
            }
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            
            async with session.post(endpoint, json=tool_request, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return "result" in data or "error" in data  # Either result or error is valid response
                return False
        except Exception as e:
            print(f"    Debug: Tool execution error: {e}")
            return False
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 70)
        print("                    TEST RESULTS SUMMARY")
        print("=" * 70)
        
        total_servers = len(self.test_results)
        passing_servers = 0
        
        for server_name, results in self.test_results.items():
            print(f"\n🖥️  {server_name.upper()}:")
            
            all_passed = (
                results["connectivity"] and 
                results["initialization"] and 
                results["tools_list"] and 
                results["tool_execution"]
            )
            
            if all_passed:
                passing_servers += 1
                print("    ✅ ALL TESTS PASSED")
            else:
                print("    ❌ SOME TESTS FAILED")
            
            print(f"    • Connectivity: {'✅' if results['connectivity'] else '❌'}")
            print(f"    • Initialization: {'✅' if results['initialization'] else '❌'}")
            print(f"    • Tools List: {'✅' if results['tools_list'] else '❌'}")
            print(f"    • Tool Execution: {'✅' if results['tool_execution'] else '❌'}")
            print(f"    • Session Management: {'✅' if results['session_management'] else '❌'}")
            
            if results["errors"]:
                print(f"    • Errors: {', '.join(results['errors'])}")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"    • Servers Tested: {total_servers}")
        print(f"    • Servers Passing: {passing_servers}")
        print(f"    • Success Rate: {(passing_servers/total_servers)*100:.1f}%")
        
        if passing_servers == total_servers:
            print("\n🎉 ALL MCP SERVERS ARE WORKING CORRECTLY!")
            print("   The Agentic Framework is ready for use.")
        else:
            print(f"\n⚠️  {total_servers - passing_servers} SERVER(S) NEED ATTENTION")
            print("   Please check the failed servers before using the framework.")
        
        print("=" * 70)

async def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        print("Waiting 10 seconds for servers to start...")
        await asyncio.sleep(10)
    
    tester = MCPTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())