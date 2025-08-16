# tests/conftest.py
import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic.core.config import Settings
from agentic.core.registry import Registry
from agentic.core.types import AgentSpec, ToolSpec, WorkflowSpec, DAG, Node


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture  
def test_settings(temp_dir):
    """Create test settings with temporary directories"""
    return Settings(
        agents_dir=str(temp_dir / "agents"),
        prompts_dir=str(temp_dir / "prompts"),
        workflows_dir=str(temp_dir / "workflows"),
        database_url=f"sqlite:///{temp_dir}/test.db",
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://test.openai.azure.com",
        auth_token="test-token"
    )


@pytest.fixture
def mock_agent():
    """Create a mock agent specification"""
    return AgentSpec(
        name="test_agent",
        endpoint="http://localhost:9999",
        description="Test agent",
        tools=[
            ToolSpec(
                name="test_tool",
                description="Test tool",
                params_schema={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                },
                returns_schema={"type": "string"}
            )
        ]
    )


@pytest.fixture
def mock_workflow():
    """Create a mock workflow specification"""
    return WorkflowSpec(
        id="test_workflow",
        name="Test Workflow",
        description="Test workflow description",
        intent="test",
        plan=DAG(nodes=[
            Node(
                id="step1",
                agent="test_agent",
                tool="test_tool",
                params={"input": "test"}
            )
        ])
    )


@pytest.fixture
def mock_registry(test_settings, mock_agent, mock_workflow, temp_dir):
    """Create a mock registry with test data"""
    # Create test agent file
    agents_dir = temp_dir / "agents"
    agents_dir.mkdir(exist_ok=True)
    
    agent_file = agents_dir / "test_agent.yaml"
    agent_file.write_text(f"""
name: {mock_agent.name}
endpoint: {mock_agent.endpoint}
description: {mock_agent.description}
tools:
  - name: {mock_agent.tools[0].name}
    description: {mock_agent.tools[0].description}
    params_schema:
      type: object
      properties:
        input:
          type: string
      required: [input]
    returns_schema:
      type: string
""")
    
    # Create test workflow file
    workflows_dir = temp_dir / "workflows"
    workflows_dir.mkdir(exist_ok=True)
    
    workflow_file = workflows_dir / "test_workflow.yaml"
    workflow_file.write_text(f"""
id: {mock_workflow.id}
name: {mock_workflow.name}
description: {mock_workflow.description}
intent: {mock_workflow.intent}
plan:
  nodes:
    - id: step1
      agent: test_agent
      tool: test_tool
      params:
        input: test
""")
    
    # Create test prompt file
    prompts_dir = temp_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    
    prompt_file = prompts_dir / "test_prompt.md"
    prompt_file.write_text("Test prompt content")
    
    # Create registry with test settings
    registry = Registry()
    registry.agents_dir = agents_dir
    registry.workflows_dir = workflows_dir  
    registry.prompts_dir = prompts_dir
    registry._load_all()
    
    return registry


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client"""
    mock_client = Mock()
    mock_client.generate = AsyncMock(return_value="Test LLM response")
    return mock_client


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client"""
    mock_client = Mock()
    mock_client.call_tool = AsyncMock(return_value={
        "success": True,
        "data": "Test tool response"
    })
    mock_client.list_tools = AsyncMock(return_value=[])
    return mock_client

