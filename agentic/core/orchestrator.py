# ============================================================================
# agentic/core/orchestrator.py
import asyncio
import time
import uuid
from typing import Dict, Any, List, Set, Optional
from collections import defaultdict, deque
import logging
from .types import DAG, Node, StepResult, ExecutionStatus
from .mcp_client import MCPClientManager
from .registry import registry

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.mcp_client = MCPClientManager()
        self.active_executions: Dict[str, Any] = {}
    
    async def execute_dag(self, dag: DAG, trace_id: Optional[str] = None) -> List[StepResult]:
        """Execute DAG with proper dependency resolution and parallel execution"""
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        logger.info(f"Starting DAG execution with trace_id: {trace_id}")
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(dag.nodes)
        parallel_groups = self._group_parallel_nodes(dag.nodes)
        
        # Track execution state
        completed_nodes: Set[str] = set()
        completed_groups: Set[str] = set()
        results: Dict[str, StepResult] = {}
        all_results: List[StepResult] = []
        
        # Execute nodes in topological order
        ready_queue = deque()
        
        # Find initial ready nodes (no dependencies)
        for node in dag.nodes:
            if not node.depends_on:
                ready_queue.append(node)
        
        while ready_queue or any(not result.success for result in all_results if not result.success):
            if not ready_queue:
                break
            
            # Group ready nodes by parallel group
            current_batch = []
            processed_groups = set()
            
            while ready_queue:
                node = ready_queue.popleft()
                
                # Skip if parallel group already processed in this batch
                if node.parallel_group and node.parallel_group in processed_groups:
                    continue
                
                if node.parallel_group:
                    # Add all nodes in parallel group
                    group_nodes = parallel_groups.get(node.parallel_group, [node])
                    current_batch.extend(group_nodes)
                    processed_groups.add(node.parallel_group)
                else:
                    current_batch.append(node)
            
            # Execute current batch in parallel
            if current_batch:
                batch_results = await self._execute_batch(current_batch, results, trace_id)
                
                for result in batch_results:
                    results[result.node_id] = result
                    all_results.append(result)
                    
                    if result.success:
                        completed_nodes.add(result.node_id)
                        
                        # Mark parallel group as completed if all nodes in group are done
                        node = next(n for n in dag.nodes if n.id == result.node_id)
                        if node.parallel_group:
                            group_nodes = parallel_groups.get(node.parallel_group, [])
                            if all(gn.id in completed_nodes for gn in group_nodes):
                                completed_groups.add(node.parallel_group)
                
                # Find next ready nodes
                for node in dag.nodes:
                    if node.id not in completed_nodes and node not in ready_queue:
                        dependencies_met = True
                        
                        for dep in node.depends_on:
                            if dep in completed_groups or dep in completed_nodes:
                                continue
                            else:
                                dependencies_met = False
                                break
                        
                        if dependencies_met:
                            ready_queue.append(node)
        
        logger.info(f"DAG execution completed with {len(all_results)} steps")
        return all_results
    
    def _build_dependency_graph(self, nodes: List[Node]) -> Dict[str, List[str]]:
        """Build dependency graph from nodes"""
        graph = defaultdict(list)
        for node in nodes:
            for dep in node.depends_on:
                graph[dep].append(node.id)
        return dict(graph)
    
    def _group_parallel_nodes(self, nodes: List[Node]) -> Dict[str, List[Node]]:
        """Group nodes by parallel group"""
        groups = defaultdict(list)
        for node in nodes:
            if node.parallel_group:
                groups[node.parallel_group].append(node)
        return dict(groups)
    
    async def _execute_batch(self, nodes: List[Node], previous_results: Dict[str, StepResult], trace_id: str) -> List[StepResult]:
        """Execute a batch of nodes in parallel"""
        tasks = []
        for node in nodes:
            task = asyncio.create_task(self._execute_node(node, previous_results, trace_id))
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_node(self, node: Node, previous_results: Dict[str, StepResult], trace_id: str) -> StepResult:
        """Execute a single node"""
        start_time = time.time()
        
        try:
            # Get agent
            agent = registry.get_agent(node.agent)
            if not agent:
                raise ValueError(f"Agent '{node.agent}' not found in registry")
            
            # Resolve parameters with template substitution
            resolved_params = self._resolve_params(node.params, previous_results)
            
            # Execute tool
            result = await self.mcp_client.call_tool(
                agent=agent,
                tool_name=node.tool,
                params=resolved_params,
                timeout=node.timeout_sec
            )
            
            return StepResult(
                node_id=node.id,
                success=result["success"],
                data=result.get("data"),
                error=result.get("error"),
                started_at=start_time,
                finished_at=time.time(),
                trace_id=trace_id,
                agent=node.agent,
                tool=node.tool
            )
            
        except Exception as e:
            logger.error(f"Node execution failed for {node.id}: {e}")
            return StepResult(
                node_id=node.id,
                success=False,
                error=str(e),
                started_at=start_time,
                finished_at=time.time(),
                trace_id=trace_id,
                agent=node.agent,
                tool=node.tool
            )
    
    def _resolve_params(self, params: Dict[str, Any], results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Resolve parameter templates with previous results"""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and "{{" in value and "}}" in value:
                # Simple template resolution
                resolved_value = value
                for result_id, result in results.items():
                    if result.success and result.data:
                        template = f"{{{{results.{result_id}.data}}}}"
                        if template in resolved_value:
                            resolved_value = resolved_value.replace(template, str(result.data))
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved

orchestrator = Orchestrator()

