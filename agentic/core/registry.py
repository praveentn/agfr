# ============================================================================
# agentic/core/registry.py
import os
import yaml
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging
from .types import AgentSpec, ToolSpec, WorkflowSpec
from .config import settings

logger = logging.getLogger(__name__)

class Registry:
    def __init__(self):
        self.agents: Dict[str, AgentSpec] = {}
        self.workflows: Dict[str, WorkflowSpec] = {}
        self.prompts: Dict[str, str] = {}
        self._load_all()
    
    def _load_all(self):
        """Load all agents, workflows, and prompts from filesystem"""
        self._load_agents()
        self._load_workflows()
        self._load_prompts()
    
    def _load_agents(self):
        """Load agent specifications from YAML files (agents directory only)"""
        agents_dir = Path(settings.agents_dir)
        if not agents_dir.exists():
            agents_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for file_path in agents_dir.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    
                    # Only load files that have required agent fields
                    if 'name' not in data or 'endpoint' not in data:
                        logger.debug(f"Skipping non-agent file: {file_path}")
                        continue
                    
                    # Skip workflow files that might be in wrong directory
                    if 'id' in data and 'plan' in data:
                        logger.warning(f"Workflow file found in agents directory, please move to workflows/: {file_path}")
                        continue
                    
                    agent = AgentSpec(**data)
                    self.agents[agent.name] = agent
                    logger.info(f"Loaded agent: {agent.name}")
            except Exception as e:
                logger.error(f"Failed to load agent from {file_path}: {e}")
    
    def _load_workflows(self):
        """Load workflow specifications from YAML files (workflows directory only)"""
        workflows_dir = Path(settings.workflows_dir)
        if not workflows_dir.exists():
            workflows_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for file_path in workflows_dir.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    
                    # Only load files that have required workflow fields
                    if 'id' not in data or 'plan' not in data:
                        logger.debug(f"Skipping non-workflow file: {file_path}")
                        continue
                    
                    workflow = WorkflowSpec(**data)
                    self.workflows[workflow.id] = workflow
                    logger.info(f"Loaded workflow: {workflow.id}")
            except Exception as e:
                logger.error(f"Failed to load workflow from {file_path}: {e}")
    
    def _load_prompts(self):
        """Load prompt templates from files"""
        prompts_dir = Path(settings.prompts_dir)
        if not prompts_dir.exists():
            prompts_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for file_path in prompts_dir.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    prompt_name = str(file_path.relative_to(prompts_dir)).replace('\\', '/').replace('.md', '')
                    self.prompts[prompt_name] = content
                    logger.info(f"Loaded prompt: {prompt_name}")
            except Exception as e:
                logger.error(f"Failed to load prompt from {file_path}: {e}")
    
    def list_agents(self) -> List[AgentSpec]:
        return list(self.agents.values())
    
    def get_agent(self, name: str) -> Optional[AgentSpec]:
        return self.agents.get(name)
    
    def list_tools(self, agent_name: Optional[str] = None) -> List[ToolSpec]:
        if agent_name:
            agent = self.agents.get(agent_name)
            return agent.tools if agent else []
        
        all_tools = []
        for agent in self.agents.values():
            all_tools.extend(agent.tools)
        return all_tools
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowSpec]:
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[WorkflowSpec]:
        return list(self.workflows.values())
    
    def get_prompt(self, name: str) -> Optional[str]:
        return self.prompts.get(name)
    
    def reload(self):
        """Reload all configurations"""
        self.agents.clear()
        self.workflows.clear()
        self.prompts.clear()
        self._load_all()

# Global registry instance
registry = Registry()
