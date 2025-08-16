# tests/integration/test_full_workflow.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from agentic.core.types import QueryRequest
from agentic.core.planner import planner
from agentic.core.orchestrator import orchestrator


class TestFullWorkflow:
    
    @pytest.mark.asyncio
    async def test_market_research_workflow(self, mock_registry, mock_mcp_client):
        """Test full market research workflow execution"""
        
        # Mock MCP client responses
        search_responses = [
            {
                "success": True,
                "data": '{"items": [{"title": "Shoe Market Report", "snippet": "Market growing at 5%", "url": "http://example.com/1"}]}'
            },
            {
                "success": True, 
                "data": '{"items": [{"title": "Brand Analysis", "snippet": "Nike leads with 30%", "url": "http://example.com/2"}]}'
            }
        ]
        
        summarize_response = {
            "success": True,
            "data": '{"summary": "The shoe market is growing steadily with Nike leading.", "key_points": ["Growth at 5%", "Nike dominance"]}'
        }
        
        tabulate_response = {
            "success": True,
            "data": '{"table": [{"brand": "Nike", "market_share": "30%", "growth": "5%"}], "format": "json"}'
        }
        
        call_count = 0
        async def mock_call_tool(agent, tool_name, params, timeout=None):
            nonlocal call_count
            if tool_name == "search":
                response = search_responses[min(call_count, len(search_responses) - 1)]
                call_count += 1
                return response
            elif tool_name == "summarize":
                return summarize_response
            elif tool_name == "tabulate":
                return tabulate_response
            else:
                return {"success": False, "error": "Unknown tool"}
        
        mock_mcp_client.call_tool = mock_call_tool
        
        # Patch dependencies
        with patch('agentic.core.orchestrator.registry', mock_registry), \
             patch('agentic.core.planner.registry', mock_registry):
            
            orchestrator.mcp_client = mock_mcp_client
            
            # Create request
            request = QueryRequest(
                text="Do market research on shoes",
                options={},
                context={}
            )
            
            # Execute workflow
            dag = await planner.create_plan(request)
            assert dag.metadata["intent"] == "market_research"
            
            results = await orchestrator.execute_dag(dag, "integration_test")
            
            # Verify results
            assert len(results) >= 2  # At least search and one processing step
            assert all(r.success for r in results), f"Failed results: {[r.error for r in results if not r.success]}"
    
    @pytest.mark.asyncio
    async def test_simple_search_workflow(self, mock_registry, mock_mcp_client):
        """Test simple search workflow execution"""
        
        # Mock search response
        mock_mcp_client.call_tool = AsyncMock(return_value={
            "success": True,
            "data": '{"items": [{"title": "Python Tutorial", "snippet": "Learn Python basics", "url": "http://example.com/python"}]}'
        })
        
        with patch('agentic.core.orchestrator.registry', mock_registry), \
             patch('agentic.core.planner.registry', mock_registry):
            
            orchestrator.mcp_client = mock_mcp_client
            
            request = QueryRequest(
                text="Search for Python tutorials",
                options={},
                context={}
            )
            
            dag = await planner.create_plan(request)
            assert dag.metadata["intent"] == "web_search"
            
            results = await orchestrator.execute_dag(dag, "search_test")
            
            assert len(results) == 1
            assert results[0].success is True
            assert results[0].tool == "search"
    
    @pytest.mark.asyncio
    async def test_predefined_workflow_execution(self, mock_registry, mock_mcp_client):
        """Test execution of predefined workflow"""
        
        mock_mcp_client.call_tool = AsyncMock(return_value={
            "success": True,
            "data": "test response"
        })
        
        with patch('agentic.core.orchestrator.registry', mock_registry), \
             patch('agentic.core.planner.registry', mock_registry):
            
            orchestrator.mcp_client = mock_mcp_client
            
            request = QueryRequest(
                text="Run test workflow",
                options={"use_workflow": "test_workflow"},
                context={}
            )
            
            dag = await planner.create_plan(request)
            
            # Should use predefined workflow
            assert len(dag.nodes) == 1
            assert dag.nodes[0].agent == "test_agent"
            assert dag.nodes[0].tool == "test_tool"
            
            results = await orchestrator.execute_dag(dag, "predefined_test")
            
            assert len(results) == 1
            assert results[0].success is True


