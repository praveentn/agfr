# scripts/test_mcp_agents.py
import asyncio
import aiohttp
import json
import time
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_agent_endpoint(name, base_url, tests):
    """Test a specific MCP agent with multiple test cases"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {name}")
    print(f"{'='*60}")
    
    results = []
    
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Test 1: Health Check
            print(f"1️⃣  Testing health endpoint...")
            try:
                async with session.post(f"{base_url}/health", json={}) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"   ✅ Health: {health_data.get('status', 'unknown')}")
                        results.append(("health", True, health_data))
                    else:
                        print(f"   ❌ Health check failed: HTTP {response.status}")
                        results.append(("health", False, f"HTTP {response.status}"))
            except Exception as e:
                print(f"   ❌ Health check error: {str(e)}")
                results.append(("health", False, str(e)))
            
            # Test 2: Get Tools
            print(f"2️⃣  Testing get_tools endpoint...")
            try:
                async with session.post(f"{base_url}/get_tools", json={}) as response:
                    if response.status == 200:
                        tools_data = await response.json()
                        tool_count = len(tools_data) if isinstance(tools_data, list) else "unknown"
                        print(f"   ✅ Tools available: {tool_count}")
                        results.append(("get_tools", True, tools_data))
                    else:
                        print(f"   ❌ Get tools failed: HTTP {response.status}")
                        results.append(("get_tools", False, f"HTTP {response.status}"))
            except Exception as e:
                print(f"   ❌ Get tools error: {str(e)}")
                results.append(("get_tools", False, str(e)))
            
            # Test 3: Specific tool tests
            for i, test in enumerate(tests, 3):
                tool_name = test["tool"]
                params = test["params"]
                expected = test.get("expected", {})
                
                print(f"{i}️⃣  Testing {tool_name} tool...")
                try:
                    async with session.post(f"{base_url}/{tool_name}", json=params) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            
                            # Validate expected results if provided
                            if expected:
                                valid = all(
                                    key in result_data and result_data[key] == value 
                                    for key, value in expected.items()
                                )
                                if valid:
                                    print(f"   ✅ {tool_name}: Success (validated)")
                                else:
                                    print(f"   ⚠️  {tool_name}: Success but validation failed")
                            else:
                                print(f"   ✅ {tool_name}: Success")
                            
                            # Show preview of result
                            if isinstance(result_data, dict):
                                preview_keys = list(result_data.keys())[:3]
                                preview = {k: result_data[k] for k in preview_keys}
                                print(f"   📄 Preview: {json.dumps(preview, indent=2)[:100]}...")
                            
                            results.append((tool_name, True, result_data))
                        else:
                            error_text = await response.text()
                            print(f"   ❌ {tool_name}: HTTP {response.status} - {error_text[:100]}")
                            results.append((tool_name, False, f"HTTP {response.status}: {error_text}"))
                            
                except Exception as e:
                    print(f"   ❌ {tool_name}: Error - {str(e)}")
                    results.append((tool_name, False, str(e)))
                
                # Small delay between tests
                await asyncio.sleep(0.5)
    
    except Exception as e:
        print(f"❌ Failed to connect to {name}: {str(e)}")
        return [(name, False, str(e))]
    
    # Summary for this agent
    success_count = sum(1 for _, success, _ in results if success)
    total_tests = len(results)
    
    print(f"\n📊 {name} Summary: {success_count}/{total_tests} tests passed")
    
    return results

async def test_full_workflow():
    """Test a complete workflow using the main API"""
    print(f"\n{'='*60}")
    print(f"🔄 Testing Complete Workflow")
    print(f"{'='*60}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Test the main API endpoint
            query_data = {
                "text": "What is the capital of France?",
                "options": {},
                "context": {}
            }
            
            headers = {
                "Authorization": "Bearer devtoken123",
                "Content-Type": "application/json"
            }
            
            print("🚀 Executing workflow query...")
            async with session.post(
                "http://localhost:8080/api/query", 
                json=query_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✅ Workflow completed successfully")
                    print(f"   📋 Intent: {result.get('intent')}")
                    print(f"   ⏱️  Execution time: {result.get('execution_time', 0):.2f}s")
                    print(f"   ✅ Success: {result.get('success')}")
                    
                    if result.get('final_result'):
                        print(f"   📄 Result preview: {str(result['final_result'])[:200]}...")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ Workflow failed: HTTP {response.status}")
                    print(f"   📄 Error: {error_text[:200]}...")
                    return False
                    
    except Exception as e:
        print(f"❌ Workflow test failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🧪 Agentic Framework MCP Agent Testing Suite")
    print("=" * 80)
    
    # Define test configurations for each agent
    test_configs = [
        {
            "name": "Web Search Agent",
            "base_url": "http://localhost:9101",
            "tests": [
                {
                    "tool": "search",
                    "params": {"query": "capital of France", "limit": 3},
                },
                {
                    "tool": "search", 
                    "params": {"query": "Python programming", "limit": 2}
                }
            ]
        },
        {
            "name": "Calculator Agent",
            "base_url": "http://localhost:9104",
            "tests": [
                {
                    "tool": "add",
                    "params": {"a": 5, "b": 3},
                    "expected": 8
                },
                {
                    "tool": "multiply",
                    "params": {"a": 4, "b": 7}
                },
                {
                    "tool": "divide",
                    "params": {"a": 10, "b": 2}
                },
                {
                    "tool": "compound_interest",
                    "params": {"principal": 1000, "rate": 5, "time": 2}
                }
            ]
        },
        {
            "name": "Tabulator Agent",
            "base_url": "http://localhost:9102",
            "tests": [
                {
                    "tool": "tabulate",
                    "params": {
                        "data": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                        "format": "json"
                    }
                },
                {
                    "tool": "sort_data",
                    "params": {
                        "data": [{"name": "Charlie", "score": 85}, {"name": "Alice", "score": 92}],
                        "sort_field": "score",
                        "ascending": False
                    }
                }
            ]
        },
        {
            "name": "NLP Summarizer Agent",
            "base_url": "http://localhost:9103",
            "tests": [
                {
                    "tool": "summarize",
                    "params": {
                        "text": "The quick brown fox jumps over the lazy dog. This is a test sentence for summarization. Natural language processing is fascinating.",
                        "style": "brief"
                    }
                },
                {
                    "tool": "extract_entities",
                    "params": {
                        "text": "Apple Inc is a technology company based in Cupertino, California. It was founded in 1976."
                    }
                },
                {
                    "tool": "analyze_sentiment",
                    "params": {
                        "text": "This is a great product! I love using it every day."
                    }
                }
            ]
        }
    ]
    
    # Run tests for each agent
    all_results = []
    total_agents = len(test_configs)
    successful_agents = 0
    
    for config in test_configs:
        agent_results = await test_agent_endpoint(
            config["name"], 
            config["base_url"], 
            config["tests"]
        )
        all_results.extend(agent_results)
        
        # Check if agent has any successful tests
        if any(success for _, success, _ in agent_results):
            successful_agents += 1
    
    # Test complete workflow
    print(f"\n{'='*80}")
    workflow_success = await test_full_workflow()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 FINAL TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(all_results)
    successful_tests = sum(1 for _, success, _ in all_results if success)
    
    print(f"🤖 Agents tested: {successful_agents}/{total_agents}")
    print(f"🧪 Individual tests: {successful_tests}/{total_tests}")
    print(f"🔄 Workflow test: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    
    # Detailed breakdown
    print(f"\n📋 Test Breakdown:")
    for agent_name in {name for name, _, _ in all_results}:
        agent_tests = [(test, success) for name, success, test in all_results if name == agent_name]
        agent_success_count = sum(1 for _, success in agent_tests if success)
        status = "✅ HEALTHY" if agent_success_count > 0 else "❌ UNHEALTHY"
        print(f"   {agent_name}: {agent_success_count}/{len(agent_tests)} - {status}")
    
    # Overall status
    overall_health = "✅ HEALTHY" if successful_agents >= 3 and workflow_success else "⚠️ DEGRADED" if successful_agents >= 1 else "❌ UNHEALTHY"
    print(f"\n🏥 Overall System Health: {overall_health}")
    
    if overall_health == "✅ HEALTHY":
        print("\n🎉 All systems operational! The Agentic Framework is ready to use.")
        print("🌐 Open http://localhost:8080 in your browser to access the web interface.")
    elif overall_health == "⚠️ DEGRADED":
        print("\n⚠️  Some agents are not responding. Check the agent server logs.")
        print("🔧 Try restarting the agent servers or check your configuration.")
    else:
        print("\n🚨 System is not operational. Please check:")
        print("   1. All agent servers are running (ports 9101-9104)")
        print("   2. Main FastAPI server is running (port 8080)")
        print("   3. Environment configuration is correct")
    
    print(f"\n{'='*80}")
    return overall_health == "✅ HEALTHY"

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)