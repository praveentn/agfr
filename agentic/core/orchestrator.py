# agentic/core/orchestrator.py
import asyncio
import time
import uuid
import re
import json
from typing import Dict, Any, List, Set, Optional
from collections import defaultdict, deque
import logging
from .types import DAG, Node, StepResult, ExecutionStatus
from .mcp_client import MCPClientManager
from .registry import registry

logger = logging.getLogger(__name__)

class DataExtractor:
    """Helper class to extract and transform data between workflow steps"""
    
    @staticmethod
    def extract_numbers(data: Any) -> List[float]:
        """Extract all numbers from various data formats"""
        numbers = []
        
        if isinstance(data, (int, float)):
            return [float(data)]
        
        if isinstance(data, str):
            # Extract numbers from text using regex
            number_patterns = [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # Regular numbers with commas
                r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?\b',  # Currency
                r'\d+(?:\.\d+)?\s*(?:dollars?|USD|\$)',  # Currency with text
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, data, re.IGNORECASE)
                for match in matches:
                    # Clean the match
                    clean_num = re.sub(r'[^\d.]', '', match)
                    if clean_num:
                        try:
                            numbers.append(float(clean_num))
                        except ValueError:
                            continue
        
        elif isinstance(data, dict):
            # Extract from dictionary
            if "numbers" in data:
                numbers.extend(DataExtractor.extract_numbers(data["numbers"]))
            elif "entities" in data and isinstance(data["entities"], dict):
                # Extract from entity extraction results
                entity_numbers = data["entities"].get("numbers", [])
                for num_str in entity_numbers:
                    numbers.extend(DataExtractor.extract_numbers(num_str))
            else:
                # Recursively search all values
                for value in data.values():
                    numbers.extend(DataExtractor.extract_numbers(value))
        
        elif isinstance(data, list):
            for item in data:
                numbers.extend(DataExtractor.extract_numbers(item))
        
        return numbers
    
    @staticmethod
    def find_calculation_values(data: Any, query_context: str = "") -> Dict[str, float]:
        """Find specific calculation values based on context"""
        numbers = DataExtractor.extract_numbers(data)
        
        # Smart mapping based on query context and found numbers
        context_lower = query_context.lower()
        
        result = {}
        
        if len(numbers) >= 2:
            # For sales calculations: quantity × price
            if any(word in context_lower for word in ['sales', 'revenue', 'total', 'earning', 'sold', 'price']):
                # Try to identify quantity vs price
                sorted_numbers = sorted(numbers, reverse=True)
                
                # Heuristics for sales calculations
                if 'iphones' in context_lower or 'phones' in context_lower:
                    # Likely: large number = quantity, smaller = price or vice versa
                    if len(numbers) >= 2:
                        # Look for quantity indicators
                        quantity_candidates = [n for n in numbers if n >= 10 and n <= 10000]  # Reasonable phone quantities
                        price_candidates = [n for n in numbers if n >= 50 and n <= 5000]     # Reasonable phone prices
                        
                        if quantity_candidates and price_candidates:
                            result['quantity'] = quantity_candidates[0]
                            result['price'] = price_candidates[-1]  # Often the last/highest price mentioned
                        else:
                            # Fallback to first two numbers
                            result['a'] = numbers[0]
                            result['b'] = numbers[1]
            
            # For compound interest: principal, rate, time
            elif any(word in context_lower for word in ['interest', 'compound', 'investment', 'rate']):
                if len(numbers) >= 3:
                    # Typically: principal (large), rate (small %), time (years)
                    principal_candidates = [n for n in numbers if n >= 1000]
                    rate_candidates = [n for n in numbers if 0 < n <= 100]
                    time_candidates = [n for n in numbers if 1 <= n <= 50]
                    
                    if principal_candidates:
                        result['principal'] = principal_candidates[0]
                    if rate_candidates:
                        result['rate'] = rate_candidates[0]
                    if time_candidates:
                        result['time'] = time_candidates[0]
                    
                    # Fill defaults if missing
                    result.setdefault('principal', numbers[0] if numbers else 10000)
                    result.setdefault('rate', 5.0)
                    result.setdefault('time', 1.0)
            
            # Default: use first two numbers as a and b
            if not result:
                result['a'] = numbers[0] if len(numbers) > 0 else 0
                result['b'] = numbers[1] if len(numbers) > 1 else 1
        
        elif len(numbers) == 1:
            # Single number operations
            result['a'] = numbers[0]
            result['b'] = 1  # Default multiplier
        
        else:
            # No numbers found - use defaults
            result['a'] = 0
            result['b'] = 1
        
        return result
    
    @staticmethod
    def extract_text_content(data: Any) -> str:
        """Extract readable text content from various data formats"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if "summary" in data:
                return data["summary"]
            elif "content" in data:
                return str(data["content"])
            elif "text" in data:
                return data["text"]
            else:
                # Join all string values
                text_parts = []
                for value in data.values():
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                text_parts.append(item)
                return " ".join(text_parts)
        elif isinstance(data, list):
            return " ".join(str(item) for item in data)
        else:
            return str(data)

class SystemAgent:
    """Built-in system agent for handling system queries"""
    
    def __init__(self, registry_ref, mcp_client_ref):
        self.registry = registry_ref
        self.mcp_client = mcp_client_ref
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system tools"""
        try:
            if tool_name == "list_tools":
                return await self._list_all_tools(params)
            elif tool_name == "list_agents":
                return await self._list_all_agents(params)
            elif tool_name == "list_workflows":
                return await self._list_all_workflows(params)
            elif tool_name == "system_status":
                return await self._get_system_status(params)
            elif tool_name == "help":
                return await self._show_help(params)
            elif tool_name == "extract_calculation_values":
                return await self._extract_calculation_values(params)
            else:
                return {
                    "success": False,
                    "error": f"Unknown system tool: {tool_name}"
                }
        except Exception as e:
            logger.error(f"System tool {tool_name} failed: {e}")
            return {
                "success": False,
                "error": f"System tool failed: {str(e)}"
            }
    
    async def _extract_calculation_values(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract calculation values from previous results"""
        try:
            data = params.get("data", {})
            context = params.get("context", "")
            operation = params.get("operation", "multiply")
            
            values = DataExtractor.find_calculation_values(data, context)
            
            return {
                "success": True,
                "values": values,
                "operation": operation,
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Value extraction failed: {str(e)}"
            }
    
    async def _list_all_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all available tools across all agents"""
        try:
            include_details = params.get("include_details", False)
            agents = self.registry.list_agents()
            
            all_tools = []
            total_tools = 0
            
            for agent in agents:
                if not agent.enabled:
                    continue
                    
                agent_tools = {
                    "agent_name": agent.name,
                    "agent_description": agent.description,
                    "endpoint": agent.endpoint,
                    "tools": []
                }
                
                for tool in agent.tools:
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description
                    }
                    
                    if include_details:
                        tool_info["params_schema"] = tool.params_schema
                        tool_info["returns_schema"] = tool.returns_schema
                    
                    agent_tools["tools"].append(tool_info)
                    total_tools += 1
                
                all_tools.append(agent_tools)
            
            # Add system tools
            system_tools = {
                "agent_name": "system",
                "agent_description": "Built-in system agent for framework management",
                "endpoint": "internal://system",
                "tools": [
                    {"name": "list_tools", "description": "List all available tools"},
                    {"name": "list_agents", "description": "List all available agents"},
                    {"name": "list_workflows", "description": "List all available workflows"},
                    {"name": "system_status", "description": "Get system health status"},
                    {"name": "help", "description": "Show help and capabilities"},
                    {"name": "extract_calculation_values", "description": "Extract calculation values from data"}
                ]
            }
            all_tools.append(system_tools)
            total_tools += len(system_tools["tools"])
            
            result = {
                "total_agents": len(all_tools),
                "total_tools": total_tools,
                "agents_and_tools": all_tools,
                "summary": f"Found {total_tools} tools across {len(all_tools)} agents"
            }
            
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return {"success": False, "error": str(e)}
    
    async def _list_all_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all available agents"""
        try:
            include_health = params.get("include_health", False)
            agents = self.registry.list_agents()
            
            agent_list = []
            
            for agent in agents:
                agent_info = {
                    "name": agent.name,
                    "description": agent.description,
                    "endpoint": agent.endpoint,
                    "enabled": agent.enabled,
                    "tool_count": len(agent.tools),
                    "tools": [tool.name for tool in agent.tools]
                }
                
                if include_health:
                    try:
                        health_status = await self.mcp_client.test_agent_connection(agent)
                        agent_info["health_status"] = "healthy" if health_status else "unhealthy"
                    except Exception as e:
                        agent_info["health_status"] = "error"
                        agent_info["health_error"] = str(e)
                
                agent_list.append(agent_info)
            
            # Add system agent
            system_agent_info = {
                "name": "system",
                "description": "Built-in system agent",
                "endpoint": "internal://system",
                "enabled": True,
                "tool_count": 6,
                "tools": ["list_tools", "list_agents", "list_workflows", "system_status", "help", "extract_calculation_values"],
                "health_status": "healthy"
            }
            agent_list.append(system_agent_info)
            
            result = {
                "total_agents": len(agent_list),
                "enabled_agents": len([a for a in agent_list if a.get("enabled", False)]),
                "agents": agent_list
            }
            
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return {"success": False, "error": str(e)}
    
    async def _list_all_workflows(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all available workflows"""
        try:
            workflows = self.registry.list_workflows()
            
            workflow_list = []
            for workflow in workflows:
                workflow_info = {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "intent": workflow.intent,
                    "node_count": len(workflow.plan.nodes),
                    "metadata": workflow.metadata
                }
                workflow_list.append(workflow_info)
            
            result = {
                "total_workflows": len(workflow_list),
                "workflows": workflow_list
            }
            
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_system_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system health and status"""
        try:
            agents = self.registry.list_agents()
            workflows = self.registry.list_workflows()
            
            # Quick health check of agents
            healthy_agents = 0
            unhealthy_agents = 0
            
            for agent in agents:
                if not agent.enabled:
                    continue
                try:
                    health = await self.mcp_client.test_agent_connection(agent)
                    if health:
                        healthy_agents += 1
                    else:
                        unhealthy_agents += 1
                except:
                    unhealthy_agents += 1
            
            system_status = {
                "status": "operational" if healthy_agents > 0 else "degraded",
                "timestamp": time.time(),
                "agents": {
                    "total": len(agents),
                    "enabled": len([a for a in agents if a.enabled]),
                    "healthy": healthy_agents,
                    "unhealthy": unhealthy_agents
                },
                "workflows": {
                    "total": len(workflows),
                    "available": len([w for w in workflows if w.intent])
                },
                "framework_version": "1.0.0"
            }
            
            return {"success": True, "data": system_status}
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"success": False, "error": str(e)}
    
    async def _show_help(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Show help and capabilities"""
        try:
            query = params.get("query", "")
            
            help_content = {
                "framework_name": "Agentic Framework",
                "version": "1.0.0",
                "description": "Multi-agent workflow orchestration system with MCP protocol support",
                "capabilities": [
                    "Market research and competitive analysis",
                    "Web search and information retrieval", 
                    "Data analysis and summarization",
                    "Mathematical calculations and statistics",
                    "Natural language processing",
                    "Data tabulation and formatting"
                ],
                "available_intents": [
                    "market_research - Perform comprehensive market analysis",
                    "web_search - Search for information online",
                    "data_analysis - Analyze and process data",
                    "calculation - Perform mathematical operations",
                    "system_query - Get information about the system"
                ],
                "example_queries": [
                    "Do market research on electric vehicles",
                    "Search for Python best practices",
                    "Calculate compound interest for $10000 at 7% for 5 years",
                    "Analyze this data and create a summary",
                    "List available tools",
                    "What agents are available?"
                ],
                "system_commands": [
                    "list tools - Show all available tools",
                    "list agents - Show all available agents",
                    "list workflows - Show all available workflows",
                    "system status - Check system health",
                    "help - Show this help message"
                ]
            }
            
            if query:
                help_content["your_query"] = query
                help_content["suggestion"] = "Try asking me to perform market research, search for information, or analyze data!"
            
            return {"success": True, "data": help_content}
            
        except Exception as e:
            logger.error(f"Failed to show help: {e}")
            return {"success": False, "error": str(e)}

class Orchestrator:
    def __init__(self):
        self.mcp_client = MCPClientManager()
        self.system_agent = SystemAgent(registry, self.mcp_client)
        self.active_executions: Dict[str, Any] = {}
        self.max_execution_time = 60  # Reduced to 1 minute for faster feedback
        self.max_node_time = 15  # Reduced to 15 seconds per node
        self.data_extractor = DataExtractor()
    
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
        
        # Check agent availability (skip for system agents)
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
        
        # Store original query context for parameter resolution
        query_context = dag.metadata.get("query", "")
        
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
                batch_results = await self._execute_batch_with_monitoring(current_batch, results, trace_id, query_context)
                
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
    
    def _resolve_params(self, params: Dict[str, Any], results: Dict[str, StepResult], query_context: str = "") -> Dict[str, Any]:
        """Enhanced parameter resolution with smart data extraction"""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and "{{" in value and "}}" in value:
                # Enhanced template resolution
                resolved_value = value
                
                # Handle results.group_name.combined for parallel groups
                if "{{results." in resolved_value and ".combined}}" in resolved_value:
                    import re
                    match = re.search(r'\{\{results\.(\w+)\.combined\}\}', resolved_value)
                    if match:
                        group_name = match.group(1)
                        combined_data = self._combine_group_results(group_name, results)
                        resolved_value = resolved_value.replace(f"{{{{results.{group_name}.combined}}}}", combined_data)
                
                # Handle smart parameter extraction
                elif "{{extract:" in resolved_value:
                    # Smart extraction patterns like {{extract:numbers_from:step_id}}
                    import re
                    extract_match = re.search(r'\{\{extract:(\w+)_from:(\w+)\}\}', resolved_value)
                    if extract_match:
                        extraction_type = extract_match.group(1)
                        source_step = extract_match.group(2)
                        
                        if source_step in results and results[source_step].success:
                            source_data = results[source_step].data
                            
                            if extraction_type == "numbers":
                                numbers = self.data_extractor.extract_numbers(source_data)
                                if numbers:
                                    resolved_value = str(numbers[0])  # Use first number found
                            elif extraction_type == "calculation_values":
                                calc_values = self.data_extractor.find_calculation_values(source_data, query_context)
                                if key in calc_values:
                                    resolved_value = str(calc_values[key])
                                elif key == 'a' and 'quantity' in calc_values:
                                    resolved_value = str(calc_values['quantity'])
                                elif key == 'b' and 'price' in calc_values:
                                    resolved_value = str(calc_values['price'])
                
                # Handle individual result references with smart extraction
                for result_id, result in results.items():
                    if result.success and result.data:
                        # Direct data reference
                        template = f"{{{{results.{result_id}.data}}}}"
                        if template in resolved_value:
                            data_str = str(result.data) if not isinstance(result.data, str) else result.data
                            resolved_value = resolved_value.replace(template, data_str)
                        
                        # Smart parameter extraction based on parameter name
                        smart_template = f"{{{{results.{result_id}.{key}}}}}"
                        if smart_template in resolved_value:
                            # Extract specific value based on parameter name
                            if key in ['a', 'b', 'quantity', 'price', 'principal', 'rate', 'time']:
                                calc_values = self.data_extractor.find_calculation_values(result.data, query_context)
                                if key in calc_values:
                                    resolved_value = str(calc_values[key])
                                elif key == 'a' and 'quantity' in calc_values:
                                    resolved_value = str(calc_values['quantity'])
                                elif key == 'b' and 'price' in calc_values:
                                    resolved_value = str(calc_values['price'])
                                else:
                                    # Fallback to first number
                                    numbers = self.data_extractor.extract_numbers(result.data)
                                    if numbers:
                                        resolved_value = str(numbers[0] if key == 'a' else numbers[1] if len(numbers) > 1 else 1)
                
                # Convert to appropriate type
                try:
                    if resolved_value != value:  # If template was resolved
                        # Try to convert to number if it looks like one
                        if resolved_value.replace('.', '').replace('-', '').isdigit():
                            resolved[key] = float(resolved_value) if '.' in resolved_value else int(resolved_value)
                        else:
                            resolved[key] = resolved_value
                    else:
                        resolved[key] = value
                except (ValueError, TypeError):
                    resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        # Additional smart parameter resolution for calculation tools
        if not any("{{" in str(v) for v in params.values()):
            # If no templates were used, try smart extraction for calculator operations
            if all(key in ['a', 'b'] for key in params.keys()):
                # This looks like a calculator operation, try to resolve from previous results
                for result_id, result in results.items():
                    if result.success and result.data:
                        calc_values = self.data_extractor.find_calculation_values(result.data, query_context)
                        for param_key in ['a', 'b']:
                            if param_key in params and params[param_key] is None:
                                if param_key in calc_values:
                                    resolved[param_key] = calc_values[param_key]
                                elif param_key == 'a' and 'quantity' in calc_values:
                                    resolved[param_key] = calc_values['quantity']
                                elif param_key == 'b' and 'price' in calc_values:
                                    resolved[param_key] = calc_values['price']
        
        return resolved
    
    async def _execute_batch_with_monitoring(self, nodes: List[Node], previous_results: Dict[str, StepResult], trace_id: str, query_context: str = "") -> List[StepResult]:
        """Execute a batch of nodes with detailed monitoring"""
        logger.info(f"Starting batch execution with {len(nodes)} nodes")
        
        tasks = []
        for node in nodes:
            task = asyncio.create_task(
                self._execute_node_with_monitoring(node, previous_results, trace_id, query_context),
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
    
    async def _execute_node_with_monitoring(self, node: Node, previous_results: Dict[str, StepResult], trace_id: str, query_context: str = "") -> StepResult:
        """Execute a single node with detailed monitoring and enhanced parameter resolution"""
        start_time = time.time()
        logger.info(f"Starting execution of node {node.id} ({node.agent}.{node.tool})")
        
        try:
            # Handle system agent
            if node.agent == "system":
                # Resolve parameters with template substitution
                resolved_params = self._resolve_params(node.params, previous_results, query_context)
                logger.debug(f"Node {node.id} resolved params: {resolved_params}")
                
                # Execute system tool
                result = await self.system_agent.execute_tool(node.tool, resolved_params)
                
                execution_time = time.time() - start_time
                logger.info(f"System node {node.id} completed in {execution_time:.2f}s, success: {result['success']}")
                
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
            
            # Handle regular MCP agents
            agent = registry.get_agent(node.agent)
            if not agent:
                raise ValueError(f"Agent '{node.agent}' not found in registry")
            
            # Enhanced parameter resolution with query context
            resolved_params = self._resolve_params(node.params, previous_results, query_context)
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
            
            # Check agent exists (skip validation for system agent)
            if node.agent != "system":
                if not registry.get_agent(node.agent):
                    errors.append(f"Node {node.id} references unknown agent: {node.agent}")
        
        return errors
    
    async def _check_agent_availability(self, dag: DAG):
        """Check if required agents are available"""
        required_agents = {node.agent for node in dag.nodes}
        
        for agent_name in required_agents:
            if agent_name == "system":
                logger.info(f"Agent {agent_name} availability: built-in (always available)")
                continue
                
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