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
        self.max_execution_time = 60  # Reduced to 1 minute for faster feedback
        self.max_node_time = 15  # Reduced to 15 seconds per node
    
    async def execute_dag(self, dag: DAG, trace_id: Optional[str] = None) -> List[StepResult]:
        """Execute DAG with improved error handling and debugging"""
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        logger.info(f"Starting DAG execution with trace_id: {trace_id}")
        logger.info(f"DAG contains {len(dag.nodes)} nodes: {[n.id for n in dag.nodes]}")
        
        start_time = time.time()
        
        # Validate DAG first
        validation_errors = self._validate_dag(dag)
        if validation_errors:
            logger.error(f"DAG validation failed: {validation_errors}")
            return [StepResult(
                node_id="validation",
                success=False,
                error=f"DAG validation failed: {', '.join(validation_errors)}",
                started_at=start_time,
                finished_at=time.time(),
                trace_id=trace_id,
                agent="system",
                tool="validate"
            )]
        
        # Check agent availability
        await self._check_agent_availability(dag)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(dag.nodes)
        parallel_groups = self._group_parallel_nodes(dag.nodes)
        
        logger.info(f"Dependency graph: {dependency_graph}")
        logger.info(f"Parallel groups: {list(parallel_groups.keys())}")
        
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
                logger.info(f"Node {node.id} is ready (no dependencies)")
        
        execution_round = 0
        while ready_queue:
            execution_round += 1
            logger.info(f"Starting execution round {execution_round}")
            
            # Check overall execution timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_execution_time:
                logger.error(f"DAG execution timed out after {elapsed_time:.2f} seconds")
                break
            
            # Group ready nodes by parallel group
            current_batch = []
            processed_groups = set()
            
            while ready_queue:
                node = ready_queue.popleft()
                
                # Skip if parallel group already processed in this batch
                if node.parallel_group and node.parallel_group in processed_groups:
                    logger.debug(f"Skipping {node.id} - parallel group {node.parallel_group} already in batch")
                    continue
                
                if node.parallel_group:
                    # Add all nodes in parallel group
                    group_nodes = parallel_groups.get(node.parallel_group, [node])
                    current_batch.extend(group_nodes)
                    processed_groups.add(node.parallel_group)
                    logger.info(f"Added parallel group {node.parallel_group} with {len(group_nodes)} nodes")
                else:
                    current_batch.append(node)
                    logger.info(f"Added single node {node.id}")
            
            # Execute current batch
            if current_batch:
                logger.info(f"Executing batch of {len(current_batch)} nodes: {[n.id for n in current_batch]}")
                batch_results = await self._execute_batch_with_monitoring(current_batch, results, trace_id)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch execution exception: {result}")
                        continue
                    
                    results[result.node_id] = result
                    all_results.append(result)
                    
                    if result.success:
                        completed_nodes.add(result.node_id)
                        logger.info(f"Node {result.node_id} completed successfully")
                        
                        # Mark parallel group as completed if all nodes in group are done
                        node = next((n for n in dag.nodes if n.id == result.node_id), None)
                        if node and node.parallel_group:
                            group_nodes = parallel_groups.get(node.parallel_group, [])
                            if all(gn.id in completed_nodes for gn in group_nodes):
                                completed_groups.add(node.parallel_group)
                                logger.info(f"Parallel group {node.parallel_group} completed")
                    else:
                        logger.error(f"Node {result.node_id} failed: {result.error}")
                
                # Find next ready nodes
                newly_ready = []
                for node in dag.nodes:
                    if node.id not in completed_nodes and node not in ready_queue and node not in newly_ready:
                        dependencies_met = True
                        
                        for dep in node.depends_on:
                            if dep in completed_groups or dep in completed_nodes:
                                continue
                            else:
                                dependencies_met = False
                                break
                        
                        if dependencies_met:
                            newly_ready.append(node)
                            logger.info(f"Node {node.id} is now ready (dependencies met)")
                
                ready_queue.extend(newly_ready)
            
            # Safety check to prevent infinite loops
            if execution_round > len(dag.nodes) * 2:
                logger.error(f"Execution round limit exceeded ({execution_round}), stopping")
                break
        
        execution_time = time.time() - start_time
        successful_count = len([r for r in all_results if r.success])
        logger.info(f"DAG execution completed in {execution_time:.2f}s with {len(all_results)} steps ({successful_count} successful)")
        
        return all_results
    
    def _validate_dag(self, dag: DAG) -> List[str]:
        """Validate DAG structure"""
        errors = []
        
        if not dag.nodes:
            errors.append("DAG has no nodes")
            return errors
        
        node_ids = {node.id for node in dag.nodes}
        
        for node in dag.nodes:
            # Check for invalid dependencies
            for dep in node.depends_on:
                if dep not in node_ids and dep not in [n.parallel_group for n in dag.nodes if n.parallel_group]:
                    errors.append(f"Node {node.id} depends on non-existent node/group: {dep}")
            
            # Check agent exists
            if not registry.get_agent(node.agent):
                errors.append(f"Node {node.id} references unknown agent: {node.agent}")
        
        return errors
    
    async def _check_agent_availability(self, dag: DAG):
        """Check if required agents are available"""
        required_agents = {node.agent for node in dag.nodes}
        
        for agent_name in required_agents:
            agent = registry.get_agent(agent_name)
            if agent:
                is_available = await self.mcp_client.test_agent_connection(agent)
                logger.info(f"Agent {agent_name} availability: {is_available}")
            else:
                logger.warning(f"Agent {agent_name} not found in registry")
    
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
    
    async def _execute_batch_with_monitoring(self, nodes: List[Node], previous_results: Dict[str, StepResult], trace_id: str) -> List[StepResult]:
        """Execute a batch of nodes with detailed monitoring"""
        logger.info(f"Starting batch execution with {len(nodes)} nodes")
        
        tasks = []
        for node in nodes:
            task = asyncio.create_task(
                self._execute_node_with_monitoring(node, previous_results, trace_id),
                name=f"node_{node.id}"
            )
            tasks.append(task)
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.max_node_time * 2  # Give extra time for batch
            )
            logger.info(f"Batch execution completed with {len(results)} results")
            return results
        except asyncio.TimeoutError:
            logger.error(f"Batch execution timed out")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Return partial results
            results = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    try:
                        results.append(task.result())
                    except Exception as e:
                        results.append(e)
            return results
    
    async def _execute_node_with_monitoring(self, node: Node, previous_results: Dict[str, StepResult], trace_id: str) -> StepResult:
        """Execute a single node with detailed monitoring"""
        start_time = time.time()
        logger.info(f"Starting execution of node {node.id} ({node.agent}.{node.tool})")
        
        try:
            # Get agent
            agent = registry.get_agent(node.agent)
            if not agent:
                raise ValueError(f"Agent '{node.agent}' not found in registry")
            
            # Resolve parameters with template substitution
            resolved_params = self._resolve_params(node.params, previous_results)
            logger.debug(f"Node {node.id} resolved params: {resolved_params}")
            
            # Execute tool with node-specific timeout
            node_timeout = min(node.timeout_sec or self.max_node_time, self.max_node_time)
            logger.info(f"Executing {node.agent}.{node.tool} with timeout {node_timeout}s")
            
            result = await asyncio.wait_for(
                self.mcp_client.call_tool(
                    agent=agent,
                    tool_name=node.tool,
                    params=resolved_params,
                    timeout=node_timeout
                ),
                timeout=node_timeout + 2  # Small buffer
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Node {node.id} completed in {execution_time:.2f}s, success: {result['success']}")
            
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
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Node execution timed out after {execution_time:.2f} seconds"
            logger.error(f"Node {node.id} timed out: {error_msg}")
            return StepResult(
                node_id=node.id,
                success=False,
                error=error_msg,
                started_at=start_time,
                finished_at=time.time(),
                trace_id=trace_id,
                agent=node.agent,
                tool=node.tool
            )
        except Exception as e:
            execution_time = time.time() - start_time
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
                
                # Handle results.group_name.combined for parallel groups
                if "{{results." in resolved_value and ".combined}}" in resolved_value:
                    # Extract group name
                    import re
                    match = re.search(r'\{\{results\.(\w+)\.combined\}\}', resolved_value)
                    if match:
                        group_name = match.group(1)
                        combined_data = self._combine_group_results(group_name, results)
                        resolved_value = resolved_value.replace(f"{{{{results.{group_name}.combined}}}}", combined_data)
                
                # Handle individual result references
                for result_id, result in results.items():
                    if result.success and result.data:
                        template = f"{{{{results.{result_id}.data}}}}"
                        if template in resolved_value:
                            data_str = str(result.data) if not isinstance(result.data, str) else result.data
                            resolved_value = resolved_value.replace(template, data_str)
                
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved
    
    def _combine_group_results(self, group_name: str, results: Dict[str, StepResult]) -> str:
        """Combine results from a parallel group"""
        combined_text = []
        
        for result in results.values():
            if result.success and result.data:
                if isinstance(result.data, dict):
                    # Extract items from search results
                    if "items" in result.data:
                        for item in result.data["items"]:
                            if isinstance(item, dict):
                                title = item.get("title", "")
                                snippet = item.get("snippet", "")
                                combined_text.append(f"{title}: {snippet}")
                            else:
                                combined_text.append(str(item))
                    else:
                        combined_text.append(str(result.data))
                else:
                    combined_text.append(str(result.data))
        
        return " ".join(combined_text)

orchestrator = Orchestrator()

