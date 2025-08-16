# agentic/core/types.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time

class ToolSpec(BaseModel):
    name: str
    description: str = ""
    params_schema: Dict[str, Any] = {}
    returns_schema: Dict[str, Any] = {}

class AgentSpec(BaseModel):
    name: str
    endpoint: str
    transport: str = "http"
    description: str = ""
    tools: List[ToolSpec] = []
    enabled: bool = True

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Node(BaseModel):
    id: str
    agent: str
    tool: str
    params: Dict[str, Any] = {}
    parallel_group: Optional[str] = None
    depends_on: List[str] = []
    timeout_sec: Optional[int] = None
    retries: int = 1
    description: str = ""

class DAG(BaseModel):
    nodes: List[Node]
    metadata: Dict[str, Any] = {}

class StepResult(BaseModel):
    node_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    started_at: float
    finished_at: float
    trace_id: str
    agent: str
    tool: str

class WorkflowSpec(BaseModel):
    id: str
    name: str
    description: str = ""
    intent: str
    inputs: Dict[str, Any] = {}
    plan: DAG
    outputs: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class ExecutionPlan(BaseModel):
    trace_id: str
    workflow_id: Optional[str] = None
    intent: str
    dag: DAG
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: List[StepResult] = []
    final_result: Any = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    text: str
    options: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    trace_id: str
    intent: str
    plan: DAG
    results: List[StepResult]
    final_result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None

