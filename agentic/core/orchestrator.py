# agentic/core/orchestrator.py
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict, deque

from .types import DAG, Node, StepResult, ExecutionContext, ExecutionStatus
from .mcp_client import MCPClientManager
from .config import settings

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    """Enhanced workflow orchestrator with intelligent parallel/sequential execution"""
    
    def __init__(self, mcp_client_manager: MCPClientManager):
        self.mcp_client = mcp_client_manager
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.max_parallel_nodes = settings.mcp_max_parallel
        
        # Performance tracking
        self.node_performance: Dict[str, List[float]] = defaultdict(list)
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "total_nodes_executed": 0
        }
    
    async def execute_dag(self, dag: DAG, context: ExecutionContext) -> List[StepResult]:
        """Execute DAG with intelligent parallel and sequential scheduling"""
        try:
            execution_id = context.trace_id
            start_time = time.time()
            
            # Initialize execution tracking
            self.active_executions[execution_id] = {
                "context": context,
                "start_time": start_time,
                "status": ExecutionStatus.RUNNING,
                "completed_nodes": set(),
                "running_nodes": set(),
                "failed_nodes": set(),
                "results": {},
                "node_futures": {}
            }
            
            logger.info(f"Starting DAG execution: {execution_id} with {len(dag.nodes)} nodes")
            
            # Validate DAG structure
            validation_result = self._validate_dag_structure(dag)
            if not validation_result["valid"]:
                return [self._create_error_result("dag_validation", "orchestrator", "validate", 
                                                execution_id, validation_result["error"])]
            
            # Build execution plan with dependency analysis
            execution_plan = self._build_execution_plan(dag)
            logger.debug(f"Execution plan: {execution_plan}")
            
            # Execute nodes in planned order with parallel optimization
            results = await self._execute_plan(dag, execution_plan, context)
            
            # Update execution tracking
            execution_time = time.time() - start_time
            self._update_execution_stats(results, execution_time)
            
            # Cleanup
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = ExecutionStatus.SUCCESS
                self.active_executions[execution_id]["end_time"] = time.time()
            
            logger.info(f"DAG execution completed: {execution_id} in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"DAG execution failed: {e}")
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = ExecutionStatus.FAILED
            
            return [self._create_error_result("orchestrator", "orchestrator", "execute", 
                                            context.trace_id, str(e))]
    
    def _validate_dag_structure(self, dag: DAG) -> Dict[str, Any]:
        """Validate DAG structure and dependencies"""
        try:
            if not dag.nodes:
                return {"valid": False, "error": "DAG must contain at least one node"}
            
            # Check for duplicate node IDs
            node_ids = [node.id for node in dag.nodes]
            if len(node_ids) != len(set(node_ids)):
                return {"valid": False, "error": "Duplicate node IDs found in DAG"}
            
            # Validate dependencies exist
            node_id_set = set(node_ids)
            for node in dag.nodes:
                for dep in node.depends_on:
                    if dep not in node_id_set:
                        return {"valid": False, "error": f"Node '{node.id}' depends on non-existent node '{dep}'"}
            
            # Check for cycles using DFS
            if self._has_cycles(dag):
                return {"valid": False, "error": "DAG contains cycles"}
            
            # Validate parallel groups
            parallel_groups = defaultdict(list)
            for node in dag.nodes:
                if node.parallel_group:
                    parallel_groups[node.parallel_group].append(node)
            
            # Ensure nodes in parallel groups don't depend on each other
            for group_name, group_nodes in parallel_groups.items():
                group_ids = {node.id for node in group_nodes}
                for node in group_nodes:
                    if any(dep in group_ids for dep in node.depends_on):
                        return {"valid": False, "error": f"Circular dependency in parallel group '{group_name}'"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"DAG validation failed: {str(e)}"}
    
    def _has_cycles(self, dag: DAG) -> bool:
        """Check for cycles in DAG using DFS"""
        try:
            # Build adjacency list
            graph = defaultdict(list)
            for node in dag.nodes:
                for dep in node.depends_on:
                    graph[dep].append(node.id)
            
            # DFS cycle detection
            WHITE, GRAY, BLACK = 0, 1, 2
            colors = {node.id: WHITE for node in dag.nodes}
            
            def dfs(node_id: str) -> bool:
                if colors[node_id] == GRAY:
                    return True  # Back edge found, cycle detected
                if colors[node_id] == BLACK:
                    return False  # Already processed
                
                colors[node_id] = GRAY
                for neighbor in graph[node_id]:
                    if dfs(neighbor):
                        return True
                colors[node_id] = BLACK
                return False
            
            for node in dag.nodes:
                if colors[node.id] == WHITE:
                    if dfs(node.id):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cycle detection failed: {e}")
            return True  # Assume cycles exist if detection fails
    
    def _build_execution_plan(self, dag: DAG) -> Dict[str, Any]:
        """Build intelligent execution plan with parallel optimization"""
        try:
            # Build dependency graph
            dependencies = {node.id: set(node.depends_on) for node in dag.nodes}
            dependents = defaultdict(set)
            for node in dag.nodes:
                for dep in node.depends_on:
                    dependents[dep].add(node.id)
            
            # Group nodes by parallel groups
            parallel_groups = defaultdict(list)
            standalone_nodes = []
            
            for node in dag.nodes:
                if node.parallel_group:
                    parallel_groups[node.parallel_group].append(node)
                else:
                    standalone_nodes.append(node)
            
            # Calculate execution levels (topological sort with parallel awareness)
            execution_levels = []
            remaining_nodes = {node.id: node for node in dag.nodes}
            node_dependencies = dependencies.copy()
            
            while remaining_nodes:
                # Find nodes with no remaining dependencies
                ready_nodes = []
                ready_groups = defaultdict(list)
                
                for node_id, node in remaining_nodes.items():
                    if not node_dependencies[node_id]:
                        if node.parallel_group:
                            ready_groups[node.parallel_group].append(node)
                        else:
                            ready_nodes.append(node)
                
                if not ready_nodes and not ready_groups:
                    # This should not happen if DAG is valid
                    logger.error("No ready nodes found, possible circular dependency")
                    break
                
                # Create execution level
                level = {
                    "standalone_nodes": ready_nodes,
                    "parallel_groups": dict(ready_groups),
                    "estimated_time": self._estimate_level_execution_time(ready_nodes, ready_groups)
                }
                execution_levels.append(level)
                
                # Remove processed nodes and update dependencies
                processed_nodes = set()
                for node in ready_nodes:
                    processed_nodes.add(node.id)
                for group_nodes in ready_groups.values():
                    for node in group_nodes:
                        processed_nodes.add(node.id)
                
                for node_id in processed_nodes:
                    remaining_nodes.pop(node_id)
                    for dependent in dependents[node_id]:
                        if dependent in node_dependencies:
                            node_dependencies[dependent].discard(node_id)
            
            return {
                "execution_levels": execution_levels,
                "total_levels": len(execution_levels),
                "max_parallelism": max(
                    len(level["standalone_nodes"]) + sum(len(group) for group in level["parallel_groups"].values())
                    for level in execution_levels
                ),
                "estimated_total_time": sum(level["estimated_time"] for level in execution_levels)
            }
            
        except Exception as e:
            logger.error(f"Execution plan building failed: {e}")
            return {
                "execution_levels": [],
                "total_levels": 0,
                "max_parallelism": 0,
                "estimated_total_time": 0,
                "error": str(e)
            }
    
    def _estimate_level_execution_time(self, standalone_nodes: List[Node], 
                                     parallel_groups: Dict[str, List[Node]]) -> float:
        """Estimate execution time for a level based on historical performance"""
        try:
            max_time = 0.0
            
            # Estimate standalone nodes (run in parallel)
            standalone_times = []
            for node in standalone_nodes:
                node_key = f"{node.agent}.{node.tool}"
                if node_key in self.node_performance:
                    avg_time = sum(self.node_performance[node_key]) / len(self.node_performance[node_key])
                    standalone_times.append(avg_time)
                else:
                    standalone_times.append(30.0)  # Default estimate
            
            if standalone_times:
                max_time = max(max_time, max(standalone_times))
            
            # Estimate parallel groups
            for group_name, group_nodes in parallel_groups.items():
                group_times = []
                for node in group_nodes:
                    node_key = f"{node.agent}.{node.tool}"
                    if node_key in self.node_performance:
                        avg_time = sum(self.node_performance[node_key]) / len(self.node_performance[node_key])
                        group_times.append(avg_time)
                    else:
                        group_times.append(30.0)  # Default estimate
                
                if group_times:
                    # Parallel group time is the maximum time within the group
                    max_time = max(max_time, max(group_times))
            
            return max_time
            
        except Exception as e:
            logger.warning(f"Time estimation failed: {e}")
            return 60.0  # Default fallback
    
    async def _execute_plan(self, dag: DAG, execution_plan: Dict[str, Any], 
                          context: ExecutionContext) -> List[StepResult]:
        """Execute the planned DAG with parallel optimization"""
        try:
            all_results = []
            execution_context = self.active_executions[context.trace_id]
            
            for level_idx, level in enumerate(execution_plan["execution_levels"]):
                logger.info(f"Executing level {level_idx + 1}/{execution_plan['total_levels']}")
                
                # Collect all nodes for this level
                level_nodes = []
                level_nodes.extend(level["standalone_nodes"])
                for group_nodes in level["parallel_groups"].values():
                    level_nodes.extend(group_nodes)
                
                if not level_nodes:
                    continue
                
                # Execute level with controlled parallelism
                level_results = await self._execute_level(level_nodes, execution_context, context)
                all_results.extend(level_results)
                
                # Check for failures and handle according to continue_on_failure
                failed_results = [r for r in level_results if not r.success]
                if failed_results:
                    # Check if any failed nodes should stop execution
                    critical_failures = [r for r in failed_results 
                                       if not self._find_node_by_id(dag, r.node_id).continue_on_failure]
                    
                    if critical_failures:
                        logger.error(f"Critical failures detected, stopping execution: {[r.node_id for r in critical_failures]}")
                        break
            
            return all_results
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return [self._create_error_result("plan_execution", "orchestrator", "execute", 
                                            context.trace_id, str(e))]
    
    async def _execute_level(self, nodes: List[Node], execution_context: Dict[str, Any], 
                           context: ExecutionContext) -> List[StepResult]:
        """Execute a level of nodes with controlled parallelism"""
        try:
            # Limit parallelism to avoid overwhelming the system
            semaphore = asyncio.Semaphore(min(self.max_parallel_nodes, len(nodes)))
            
            # Create execution tasks
            tasks = []
            for node in nodes:
                task = asyncio.create_task(
                    self._execute_node_with_semaphore(node, execution_context, context, semaphore)
                )
                tasks.append(task)
                execution_context["node_futures"][node.id] = task
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            level_results = []
            for i, result in enumerate(results):
                node = nodes[i]
                if isinstance(result, Exception):
                    logger.error(f"Node {node.id} execution failed with exception: {result}")
                    step_result = self._create_error_result(node.id, node.agent, node.tool, 
                                                          context.trace_id, str(result))
                else:
                    step_result = result
                
                level_results.append(step_result)
                execution_context["results"][node.id] = step_result
                
                # Update tracking
                if step_result.success:
                    execution_context["completed_nodes"].add(node.id)
                else:
                    execution_context["failed_nodes"].add(node.id)
                
                execution_context["running_nodes"].discard(node.id)
            
            return level_results
            
        except Exception as e:
            logger.error(f"Level execution failed: {e}")
            return [self._create_error_result("level_execution", "orchestrator", "execute", 
                                            context.trace_id, str(e))]
    
    async def _execute_node_with_semaphore(self, node: Node, execution_context: Dict[str, Any], 
                                         context: ExecutionContext, semaphore: asyncio.Semaphore) -> StepResult:
        """Execute a single node with semaphore control"""
        async with semaphore:
            execution_context["running_nodes"].add(node.id)
            return await self._execute_single_node(node, execution_context, context)
    
    async def _execute_single_node(self, node: Node, execution_context: Dict[str, Any], 
                                 context: ExecutionContext) -> StepResult:
        """Execute a single node with comprehensive error handling and retries"""
        start_time = time.time()
        last_error = None
        
        for attempt in range(node.retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying node {node.id}, attempt {attempt + 1}/{node.retries + 1}")
                    await asyncio.sleep(node.retry_delay * attempt)  # Exponential backoff
                
                # Prepare parameters with result substitution
                prepared_params = await self._prepare_node_parameters(node, execution_context["results"])
                
                # Set timeout
                timeout = node.timeout_sec or context.timeout_sec or settings.mcp_default_timeout_sec
                
                # Execute the tool call
                result_data = await asyncio.wait_for(
                    self.mcp_client.call_tool(node.agent, node.tool, prepared_params),
                    timeout=timeout
                )
                
                # Record performance
                execution_time = time.time() - start_time
                node_key = f"{node.agent}.{node.tool}"
                self.node_performance[node_key].append(execution_time)
                
                # Keep only last 100 performance records per node
                if len(self.node_performance[node_key]) > 100:
                    self.node_performance[node_key] = self.node_performance[node_key][-100:]
                
                return StepResult(
                    node_id=node.id,
                    agent=node.agent,
                    tool=node.tool,
                    success=True,
                    data=result_data,
                    started_at=start_time,
                    finished_at=time.time(),
                    execution_time=execution_time,
                    trace_id=context.trace_id,
                    retry_count=attempt,
                    metadata={
                        "parameters": prepared_params,
                        "timeout_used": timeout,
                        "parallel_group": node.parallel_group
                    }
                )
                
            except asyncio.TimeoutError:
                last_error = f"Node execution timed out after {timeout}s"
                logger.warning(f"Node {node.id} timed out on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Node {node.id} failed on attempt {attempt + 1}: {e}")
        
        # All retries exhausted
        execution_time = time.time() - start_time
        logger.error(f"Node {node.id} failed after {node.retries + 1} attempts: {last_error}")
        
        return StepResult(
            node_id=node.id,
            agent=node.agent,
            tool=node.tool,
            success=False,
            error=last_error,
            started_at=start_time,
            finished_at=time.time(),
            execution_time=execution_time,
            trace_id=context.trace_id,
            retry_count=node.retries,
            metadata={
                "timeout_used": node.timeout_sec or context.timeout_sec or settings.mcp_default_timeout_sec,
                "parallel_group": node.parallel_group
            }
        )
    
    async def _prepare_node_parameters(self, node: Node, previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Prepare node parameters with result substitution"""
        try:
            prepared_params = {}
            
            for key, value in node.params.items():
                if isinstance(value, str):
                    # Handle parameter substitution like {{results.node_id.data}}
                    substituted_value = value
                    
                    # Simple pattern matching for result substitution
                    import re
                    pattern = r'\{\{results\.([^.]+)\.([^}]+)\}\}'
                    matches = re.findall(pattern, value)
                    
                    for node_id, field_path in matches:
                        if node_id in previous_results:
                            result = previous_results[node_id]
                            if result.success and result.data:
                                # Navigate field path (e.g., data.summary.key_points)
                                field_value = self._get_nested_field(result.data, field_path)
                                if field_value is not None:
                                    placeholder = f"{{{{results.{node_id}.{field_path}}}}}"
                                    substituted_value = substituted_value.replace(placeholder, str(field_value))
                    
                    prepared_params[key] = substituted_value
                else:
                    prepared_params[key] = value
            
            return prepared_params
            
        except Exception as e:
            logger.error(f"Parameter preparation failed: {e}")
            return node.params.copy()  # Return original params as fallback
    
    def _get_nested_field(self, data: Any, field_path: str) -> Any:
        """Get nested field value from data using dot notation"""
        try:
            current = data
            parts = field_path.split('.')
            
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    index = int(part)
                    if 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                else:
                    return None
                
                if current is None:
                    return None
            
            return current
            
        except Exception as e:
            logger.debug(f"Field path navigation failed: {e}")
            return None
    
    def _find_node_by_id(self, dag: DAG, node_id: str) -> Optional[Node]:
        """Find node by ID in DAG"""
        for node in dag.nodes:
            if node.id == node_id:
                return node
        return None
    
    def _create_error_result(self, node_id: str, agent: str, tool: str, trace_id: str, error: str) -> StepResult:
        """Create an error step result"""
        current_time = time.time()
        return StepResult(
            node_id=node_id,
            agent=agent,
            tool=tool,
            success=False,
            error=error,
            started_at=current_time,
            finished_at=current_time,
            execution_time=0.0,
            trace_id=trace_id
        )
    
    def _update_execution_stats(self, results: List[StepResult], execution_time: float):
        """Update orchestrator execution statistics"""
        try:
            self.execution_stats["total_executions"] += 1
            self.execution_stats["total_nodes_executed"] += len(results)
            
            successful_results = [r for r in results if r.success]
            if len(successful_results) == len(results):
                self.execution_stats["successful_executions"] += 1
            else:
                self.execution_stats["failed_executions"] += 1
            
            # Update average execution time
            total_time = self.execution_stats["avg_execution_time"] * (self.execution_stats["total_executions"] - 1)
            self.execution_stats["avg_execution_time"] = (total_time + execution_time) / self.execution_stats["total_executions"]
            
            # Store execution history (keep last 1000)
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "total_nodes": len(results),
                "successful_nodes": len(successful_results),
                "failed_nodes": len(results) - len(successful_results)
            }
            
            self.execution_history.append(execution_record)
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    async def cancel_execution(self, trace_id: str) -> bool:
        """Cancel an active execution"""
        try:
            if trace_id not in self.active_executions:
                return False
            
            execution = self.active_executions[trace_id]
            execution["status"] = ExecutionStatus.CANCELLED
            
            # Cancel running node futures
            for node_id, future in execution["node_futures"].items():
                if not future.done():
                    future.cancel()
                    logger.info(f"Cancelled node: {node_id}")
            
            logger.info(f"Execution cancelled: {trace_id}")
            return True
            
        except Exception as e:
            logger.error(f"Execution cancellation failed: {e}")
            return False
    
    def get_execution_status(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get current execution status"""
        try:
            if trace_id not in self.active_executions:
                return None
            
            execution = self.active_executions[trace_id]
            
            return {
                "trace_id": trace_id,
                "status": execution["status"],
                "start_time": execution["start_time"],
                "end_time": execution.get("end_time"),
                "completed_nodes": list(execution["completed_nodes"]),
                "running_nodes": list(execution["running_nodes"]),
                "failed_nodes": list(execution["failed_nodes"]),
                "total_nodes": len(execution["context"].plan.nodes) if hasattr(execution["context"], "plan") else 0,
                "progress_percentage": len(execution["completed_nodes"]) / max(1, len(execution["context"].plan.nodes) if hasattr(execution["context"], "plan") else 1) * 100
            }
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return None
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        try:
            return {
                "execution_stats": self.execution_stats.copy(),
                "active_executions": len(self.active_executions),
                "node_performance_summary": {
                    node_key: {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
                    for node_key, times in self.node_performance.items()
                    if times
                },
                "recent_executions": self.execution_history[-10:],  # Last 10 executions
                "settings": {
                    "max_parallel_nodes": self.max_parallel_nodes,
                    "default_timeout": settings.mcp_default_timeout_sec
                }
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}
    
    async def cleanup_completed_executions(self, max_age_hours: int = 24):
        """Clean up old completed executions"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            completed_executions = []
            for trace_id, execution in self.active_executions.items():
                if execution["status"] in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                    end_time = execution.get("end_time", current_time)
                    if end_time < cutoff_time:
                        completed_executions.append(trace_id)
            
            for trace_id in completed_executions:
                del self.active_executions[trace_id]
                logger.debug(f"Cleaned up execution: {trace_id}")
            
            logger.info(f"Cleaned up {len(completed_executions)} old executions")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Create global instance
# Note: This will be initialized with MCP client manager in main application
orchestrator: Optional[WorkflowOrchestrator] = None

def initialize_orchestrator(mcp_client_manager: MCPClientManager) -> WorkflowOrchestrator:
    """Initialize the global orchestrator instance"""
    global orchestrator
    orchestrator = WorkflowOrchestrator(mcp_client_manager)
    return orchestrator