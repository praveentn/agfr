# agentic/core/types.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum

class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(str, Enum):
    """Agent type enumeration"""
    MCP_SERVER = "mcp_server"
    HTTP_API = "http_api"
    LOCAL_FUNCTION = "local_function"

class ToolSpec(BaseModel):
    """Tool specification with enhanced validation"""
    model_config = ConfigDict(
        # FIXED: Updated for Pydantic V2 - renamed from allow_population_by_field_name
        validate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True
    )
    
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    params_schema: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters JSON schema")
    returns_schema: Dict[str, Any] = Field(default_factory=dict, description="Tool return value JSON schema")
    version: Optional[str] = Field(None, description="Tool version")
    deprecated: bool = Field(False, description="Whether tool is deprecated")
    experimental: bool = Field(False, description="Whether tool is experimental")
    
    def __str__(self) -> str:
        return f"ToolSpec(name='{self.name}', version='{self.version}')"

class AgentSpec(BaseModel):
    """Agent specification with enhanced metadata"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True
    )
    
    name: str = Field(..., description="Agent name", min_length=1)
    endpoint: str = Field(..., description="Agent endpoint URL")
    description: Optional[str] = Field(None, description="Agent description")
    transport: str = Field("http", description="Transport protocol")
    agent_type: AgentType = Field(AgentType.MCP_SERVER, description="Type of agent")
    tools: List[ToolSpec] = Field(default_factory=list, description="Available tools")
    version: Optional[str] = Field(None, description="Agent version")
    health_check_url: Optional[str] = Field(None, description="Health check endpoint")
    timeout_sec: int = Field(45, ge=1, le=300, description="Default timeout in seconds")
    retries: int = Field(2, ge=0, le=5, description="Default retry count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def __str__(self) -> str:
        return f"AgentSpec(name='{self.name}', endpoint='{self.endpoint}', tools={len(self.tools)})"

class Node(BaseModel):
    """DAG node with enhanced execution control"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="forbid"
    )
    
    id: str = Field(..., description="Unique node identifier", min_length=1)
    agent: str = Field(..., description="Agent name to execute", min_length=1)
    tool: str = Field(..., description="Tool name to call", min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    parallel_group: Optional[str] = Field(None, description="Parallel execution group")
    depends_on: List[str] = Field(default_factory=list, description="Dependencies (node IDs)")
    condition: Optional[str] = Field(None, description="Execution condition")
    timeout_sec: Optional[int] = Field(None, ge=1, le=600, description="Node timeout")
    retries: int = Field(1, ge=0, le=5, description="Retry count")
    retry_delay: float = Field(1.0, ge=0.1, le=60.0, description="Retry delay in seconds")
    continue_on_failure: bool = Field(False, description="Continue workflow if this node fails")
    
    def __str__(self) -> str:
        return f"Node(id='{self.id}', agent='{self.agent}', tool='{self.tool}')"

class DAG(BaseModel):
    """Directed Acyclic Graph with validation"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="forbid"
    )
    
    id: Optional[str] = Field(None, description="DAG identifier")
    nodes: List[Node] = Field(..., description="List of nodes", min_length=1)
    edges: List[Dict[str, str]] = Field(default_factory=list, description="Explicit edges (optional)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="DAG metadata")
    max_parallel: int = Field(8, ge=1, le=20, description="Maximum parallel execution")
    timeout_sec: int = Field(300, ge=30, le=3600, description="Total DAG timeout")
    
    def validate_acyclic(self) -> bool:
        """Validate that the DAG has no cycles"""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Find node and check dependencies
            node = next((n for n in self.nodes if n.id == node_id), None)
            if node:
                for dep in node.depends_on:
                    if has_cycle(dep):
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        # Check all nodes
        for node in self.nodes:
            if node.id not in visited:
                if has_cycle(node.id):
                    return False
        return True
    
    def __str__(self) -> str:
        return f"DAG(nodes={len(self.nodes)}, edges={len(self.edges)})"

class StepResult(BaseModel):
    """Execution step result with detailed metadata"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="allow"  # Allow extra fields for tool-specific data
    )
    
    node_id: str = Field(..., description="Node identifier")
    agent: str = Field(..., description="Agent that executed")
    tool: str = Field(..., description="Tool that was called")
    success: bool = Field(..., description="Whether execution succeeded")
    data: Optional[Any] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: float = Field(..., description="Start timestamp")
    finished_at: float = Field(..., description="Finish timestamp")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    trace_id: str = Field(..., description="Trace identifier")
    retry_count: int = Field(0, description="Number of retries attempted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @property
    def duration(self) -> float:
        """Calculate execution duration"""
        return self.finished_at - self.started_at
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"StepResult({status} {self.node_id}: {self.agent}.{self.tool})"

class ExecutionPlan(BaseModel):
    """Complete execution plan with metadata"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="forbid"
    )
    
    intent: str = Field(..., description="Classified intent")
    plan: DAG = Field(..., description="Execution DAG")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Planning reasoning")
    workflow_id: Optional[str] = Field(None, description="Source workflow ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters")
    estimated_duration: Optional[float] = Field(None, description="Estimated execution time")
    created_at: datetime = Field(default_factory=datetime.now, description="Plan creation time")
    
    def __str__(self) -> str:
        return f"ExecutionPlan(intent='{self.intent}', nodes={len(self.plan.nodes)}, confidence={self.confidence})"

class QueryRequest(BaseModel):
    """Query request with options"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True
    )
    
    text: str = Field(..., description="Query text", min_length=1)
    workflow_id: Optional[str] = Field(None, description="Specific workflow to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    trace_id: Optional[str] = Field(None, description="Request trace ID")
    timeout_sec: Optional[int] = Field(None, ge=10, le=600, description="Request timeout")
    async_execution: bool = Field(False, description="Execute asynchronously")
    
    def __str__(self) -> str:
        return f"QueryRequest(text='{self.text[:50]}...', workflow='{self.workflow_id}')"

class QueryResponse(BaseModel):
    """Query response with comprehensive results"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="allow"
    )
    
    success: bool = Field(..., description="Overall success status")
    trace_id: str = Field(..., description="Execution trace ID")
    intent: str = Field(..., description="Detected intent")
    plan: Optional[ExecutionPlan] = Field(None, description="Execution plan used")
    results: List[StepResult] = Field(default_factory=list, description="Step results")
    final_result: Optional[Dict[str, Any]] = Field(None, description="Composed final result")
    error: Optional[str] = Field(None, description="Overall error message")
    started_at: float = Field(..., description="Execution start time")
    finished_at: float = Field(..., description="Execution finish time")
    execution_time: Optional[float] = Field(None, description="Total execution time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    
    @property
    def duration(self) -> float:
        """Calculate total execution duration"""
        return self.finished_at - self.started_at
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of steps"""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return (successful / len(self.results)) * 100.0
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"QueryResponse({status} {self.intent}: {len(self.results)} steps, {self.duration:.2f}s)"

class WorkflowSpec(BaseModel):
    """Enhanced workflow specification"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="forbid"
    )
    
    id: str = Field(..., description="Workflow identifier", min_length=1)
    name: str = Field(..., description="Workflow name", min_length=1)
    description: str = Field(..., description="Workflow description")
    intent: str = Field(..., description="Primary intent")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    plan: DAG = Field(..., description="Workflow execution plan")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    version: str = Field("1.0", description="Workflow version")
    enabled: bool = Field(True, description="Whether workflow is enabled")
    
    def __str__(self) -> str:
        return f"WorkflowSpec(id='{self.id}', name='{self.name}', intent='{self.intent}')"

class SystemStatus(BaseModel):
    """System status information"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="allow"
    )
    
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(..., description="Status timestamp")
    agents: Dict[str, str] = Field(default_factory=dict, description="Agent statuses")
    database: str = Field("unknown", description="Database status")
    llm_client: str = Field("unknown", description="LLM client status")
    active_executions: int = Field(0, description="Number of active executions")
    total_executions: int = Field(0, description="Total executions since start")
    uptime_seconds: float = Field(0.0, description="System uptime in seconds")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage stats")
    
    def __str__(self) -> str:
        return f"SystemStatus(status='{self.status}', agents={len(self.agents)}, uptime={self.uptime_seconds:.0f}s)"

class ExecutionContext(BaseModel):
    """Execution context for workflow runs"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="allow"
    )
    
    trace_id: str = Field(..., description="Unique trace identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier") 
    environment: str = Field("production", description="Execution environment")
    priority: int = Field(5, ge=1, le=10, description="Execution priority")
    max_parallel: int = Field(8, ge=1, le=20, description="Max parallel nodes")
    timeout_sec: int = Field(300, ge=30, le=3600, description="Total timeout")
    retry_failed: bool = Field(True, description="Retry failed nodes")
    debug_mode: bool = Field(False, description="Enable debug logging")
    created_at: datetime = Field(default_factory=datetime.now, description="Context creation time")
    
    def __str__(self) -> str:
        return f"ExecutionContext(trace_id='{self.trace_id}', priority={self.priority})"

class AgentHealth(BaseModel):
    """Agent health check result"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="allow"
    )
    
    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Health status")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check time")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    version: Optional[str] = Field(None, description="Agent version")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    
    def __str__(self) -> str:
        return f"AgentHealth(agent='{self.agent_name}', status='{self.status}')"

class DatabaseExecutionResult(BaseModel):
    """Database execution result for SQL queries"""
    model_config = ConfigDict(
        validate_by_name=True,
        extra="allow"
    )
    
    success: bool = Field(..., description="Query execution success")
    query: str = Field(..., description="Executed query")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Query result data")
    columns: List[str] = Field(default_factory=list, description="Column names")
    row_count: int = Field(0, description="Number of rows returned")
    affected_rows: Optional[int] = Field(None, description="Number of affected rows")
    execution_time: float = Field(..., description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Query warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"DatabaseResult({status} {self.row_count} rows, {self.execution_time:.3f}s)"

# Validation functions
def validate_workflow_spec(spec: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate workflow specification"""
    try:
        # Check required fields
        required_fields = ["id", "name", "description", "intent", "plan"]
        for field in required_fields:
            if field not in spec:
                return False, f"Missing required field: {field}"
        
        # Check plan structure
        plan = spec["plan"]
        if "nodes" not in plan:
            return False, "Plan must contain 'nodes' field"
        
        if not isinstance(plan["nodes"], list) or len(plan["nodes"]) == 0:
            return False, "Plan must contain at least one node"
        
        # Validate each node
        for i, node in enumerate(plan["nodes"]):
            node_fields = ["id", "agent", "tool"]
            for field in node_fields:
                if field not in node:
                    return False, f"Node {i} missing required field: {field}"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_dag_structure(dag: DAG) -> Tuple[bool, Optional[str]]:
    """Validate DAG structure and dependencies"""
    try:
        node_ids = {node.id for node in dag.nodes}
        
        # Check for duplicate node IDs
        if len(node_ids) != len(dag.nodes):
            return False, "Duplicate node IDs found"
        
        # Check dependencies exist
        for node in dag.nodes:
            for dep in node.depends_on:
                if dep not in node_ids:
                    return False, f"Node '{node.id}' depends on non-existent node '{dep}'"
        
        # Check for cycles
        if not dag.validate_acyclic():
            return False, "DAG contains cycles"
        
        return True, None
        
    except Exception as e:
        return False, f"DAG validation error: {str(e)}"