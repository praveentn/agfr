# tests/test_registry.py
import pytest
from pathlib import Path
import yaml

from agentic.core.registry import Registry
from agentic.core.types import AgentSpec, WorkflowSpec


class TestRegistry:
    
    def test_load_agents(self, mock_registry):
        """Test loading agents from YAML files"""
        agents = mock_registry.list_agents()
        assert len(agents) == 1
        
        agent = agents[0]
        assert agent.name == "test_agent"
        assert agent.endpoint == "http://localhost:9999"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"
    
    def test_load_workflows(self, mock_registry):
        """Test loading workflows from YAML files"""
        workflows = mock_registry.list_workflows()
        assert len(workflows) == 1
        
        workflow = workflows[0]
        assert workflow.id == "test_workflow"
        assert workflow.intent == "test"
        assert len(workflow.plan.nodes) == 1
    
    def test_load_prompts(self, mock_registry):
        """Test loading prompts from files"""
        prompt = mock_registry.get_prompt("test_prompt")
        assert prompt == "Test prompt content"
    
    def test_get_agent(self, mock_registry):
        """Test getting specific agent"""
        agent = mock_registry.get_agent("test_agent")
        assert agent is not None
        assert agent.name == "test_agent"
        
        # Test non-existent agent
        assert mock_registry.get_agent("nonexistent") is None
    
    def test_list_tools(self, mock_registry):
        """Test listing tools"""
        # List all tools
        all_tools = mock_registry.list_tools()
        assert len(all_tools) == 1
        assert all_tools[0].name == "test_tool"
        
        # List tools for specific agent
        agent_tools = mock_registry.list_tools("test_agent")
        assert len(agent_tools) == 1
        assert agent_tools[0].name == "test_tool"
        
        # List tools for non-existent agent
        no_tools = mock_registry.list_tools("nonexistent")
        assert len(no_tools) == 0
    
    def test_reload(self, mock_registry, temp_dir):
        """Test registry reload functionality"""
        # Add a new agent file
        agents_dir = temp_dir / "agents"
        new_agent_file = agents_dir / "new_agent.yaml"
        new_agent_file.write_text("""
name: new_agent
endpoint: http://localhost:9998
description: New test agent
tools:
  - name: new_tool
    description: New test tool
    params_schema:
      type: object
    returns_schema:
      type: string
""")
        
        # Reload registry
        mock_registry.reload()
        
        # Check new agent is loaded
        agents = mock_registry.list_agents()
        assert len(agents) == 2
        
        new_agent = mock_registry.get_agent("new_agent")
        assert new_agent is not None
        assert new_agent.name == "new_agent"


