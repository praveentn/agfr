# tests/test_orchestrator.py
import pytest
from unittest.mock import AsyncMock, Mock, patch
import time

from agentic.core.orchestrator import Orchestrator
from agentic.core.types import DAG, Node, StepResult


class TestOrchestrator:
    
    @pytest.mark.asyncio
    async def test_execute_simple_dag(self, mock_mcp_client, mock_registry):
        """Test executing a simple DAG with one node"""
        orchestrator = Orchestrator()
        orchestrator.mcp_client = mock_mcp_client
        
        # Create simple DAG
        dag = DAG(nodes=[
            Node(
                id="step1",
                agent="test_agent", 
                tool="test_tool",
                params={"input": "test"}
            )
        ])
        
        with patch('agentic.core.orchestrator.registry', mock_registry):
            results = await orchestrator.execute_dag(dag, "test_trace")
        
        assert len(results) == 1
        assert results[0].node_id == "step1"
        assert results[0].success is True
        assert results[0].trace_id == "test_trace"
    
    @pytest.mark.asyncio
    async def test_execute_parallel_dag(self, mock_mcp_client, mock_registry):
        """Test executing DAG with parallel nodes"""
        orchestrator = Orchestrator()
        orchestrator.mcp_client = mock_mcp_client
        
        # Create DAG with parallel nodes
        dag = DAG(nodes=[
            Node(
                id="step1",
                agent="test_agent",
                tool="test_tool",
                params={"input": "test1"},
                parallel_group="group_a"
            ),
            Node(
                id="step2", 
                agent="test_agent",
                tool="test_tool",
                params={"input": "test2"},
                parallel_group="group_a"
            )
        ])
        
        with patch('agentic.core.orchestrator.registry', mock_registry):
            results = await orchestrator.execute_dag(dag, "test_trace")
        
        assert len(results) == 2
        # Both should complete (parallel execution)
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_sequential_dag(self, mock_mcp_client, mock_registry):
        """Test executing DAG with sequential dependencies"""
        orchestrator = Orchestrator()
        orchestrator.mcp_client = mock_mcp_client
        
        # Create DAG with dependencies
        dag = DAG(nodes=[
            Node(
                id="step1",
                agent="test_agent",
                tool="test_tool", 
                params={"input": "test1"}
            ),
            Node(
                id="step2",
                agent="test_agent",
                tool="test_tool",
                params={"input": "test2"},
                depends_on=["step1"]
            )
        ])
        
        with patch('agentic.core.orchestrator.registry', mock_registry):
            results = await orchestrator.execute_dag(dag, "test_trace")
        
        assert len(results) == 2
        assert all(r.success for r in results)
        
        # Check execution order (step1 should finish before step2 starts)
        step1_result = next(r for r in results if r.node_id == "step1")
        step2_result = next(r for r in results if r.node_id == "step2")
        assert step1_result.finished_at <= step2_result.started_at
    
    @pytest.mark.asyncio
    async def test_handle_node_failure(self, mock_registry):
        """Test handling node execution failure"""
        orchestrator = Orchestrator()
        
        # Mock MCP client to return failure
        mock_mcp_client = Mock()
        mock_mcp_client.call_tool = AsyncMock(return_value={
            "success": False,
            "error": "Tool execution failed"
        })
        orchestrator.mcp_client = mock_mcp_client
        
        dag = DAG(nodes=[
            Node(
                id="failing_step",
                agent="test_agent",
                tool="test_tool",
                params={"input": "test"}
            )
        ])
        
        with patch('agentic.core.orchestrator.registry', mock_registry):
            results = await orchestrator.execute_dag(dag, "test_trace")
        
        assert len(results) == 1
        assert results[0].success is False
        assert "Tool execution failed" in results[0].error
    
    def test_resolve_params(self):
        """Test parameter template resolution"""
        orchestrator = Orchestrator()
        
        # Create mock previous results
        previous_results = {
            "step1": StepResult(
                node_id="step1",
                success=True,
                data="result_data",
                started_at=time.time(),
                finished_at=time.time(),
                trace_id="test",
                agent="test_agent",
                tool="test_tool"
            )
        }
        
        # Test parameter resolution
        params = {
            "static_param": "static_value",
            "dynamic_param": "{{results.step1.data}}"
        }
        
        resolved = orchestrator._resolve_params(params, previous_results)
        
        assert resolved["static_param"] == "static_value"
        assert resolved["dynamic_param"] == "result_data"

