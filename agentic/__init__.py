# agentic/__init__.py
"""
Agentic Framework - Multi-agent workflow orchestration system
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.types import (
    AgentSpec,
    ToolSpec, 
    WorkflowSpec,
    DAG,
    Node,
    StepResult,
    ExecutionPlan,
    QueryRequest,
    QueryResponse,
)

from .core.registry import registry
from .core.planner import planner
from .core.orchestrator import orchestrator
from .core.config import settings

__all__ = [
    "AgentSpec",
    "ToolSpec",
    "WorkflowSpec", 
    "DAG",
    "Node",
    "StepResult",
    "ExecutionPlan",
    "QueryRequest",
    "QueryResponse",
    "registry",
    "planner",
    "orchestrator",
    "settings",
]

