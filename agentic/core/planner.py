# agentic/core/planner.py
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import yaml
from pathlib import Path

from .types import ExecutionPlan, DAG, Node, WorkflowSpec, AgentSpec, ToolSpec
from .config import settings
from .registry import registry
from .llm_client import llm_client

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Enhanced workflow management with user-defined workflows"""
    
    def __init__(self):
        self.user_workflows: Dict[str, Dict[str, Any]] = {}
        self.load_user_workflows()
    
    def load_user_workflows(self):
        """Load user-defined workflows from storage"""
        # In a real implementation, this would load from database
        # For now, we'll maintain in-memory storage
        pass
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows (predefined + user-created)"""
        workflows = []
        
        # Add predefined workflows
        for workflow_id in registry.list_workflows():
            try:
                spec = registry.get_workflow(workflow_id)
                if spec:
                    workflows.append({
                        "id": spec.id,
                        "name": spec.name,
                        "description": spec.description,
                        "intent": spec.intent,
                        "tags": spec.tags,
                        "metadata": spec.metadata,
                        "type": "predefined",
                        "author": spec.metadata.get("author", "system")
                    })
            except Exception as e:
                logger.error(f"Error loading predefined workflow {workflow_id}: {e}")
        
        # Add user workflows
        for workflow_id, workflow in self.user_workflows.items():
            workflows.append({
                "id": workflow_id,
                "name": workflow.get("name", workflow_id),
                "description": workflow.get("description", ""),
                "intent": workflow.get("intent", "custom"),
                "tags": workflow.get("tags", []),
                "metadata": workflow.get("metadata", {}),
                "type": "user_defined",
                "author": workflow.get("metadata", {}).get("author", "user")
            })
        
        return workflows
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user-defined workflow"""
        try:
            # Validate workflow structure
            required_fields = ["name", "description", "nodes"]
            for field in required_fields:
                if field not in workflow_data:
                    return {"success": False, "error": f"Missing required field: {field}"}
            
            # Generate unique ID
            workflow_id = f"user_{workflow_data['name'].lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
            
            # Add required plan.id if missing
            if "plan" in workflow_data and "id" not in workflow_data["plan"]:
                workflow_data["plan"]["id"] = f"{workflow_id}_plan"
            
            # Create workflow spec
            workflow = {
                "id": workflow_id,
                "name": workflow_data["name"],
                "description": workflow_data["description"],
                "intent": workflow_data.get("intent", "custom"),
                "tags": workflow_data.get("tags", []),
                "plan": workflow_data.get("plan", {"id": f"{workflow_id}_plan", "nodes": workflow_data["nodes"]}),
                "metadata": {
                    **workflow_data.get("metadata", {}),
                    "author": "user",
                    "created_at": datetime.now().isoformat(),
                    "type": "user_defined"
                }
            }
            
            # Validate nodes
            validation_result = await self._validate_workflow_nodes(workflow["plan"]["nodes"])
            if not validation_result["valid"]:
                return {"success": False, "error": f"Workflow validation failed: {validation_result['error']}"}
            
            # Store workflow
            self.user_workflows[workflow_id] = workflow
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "message": f"Workflow '{workflow['name']}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a user-defined workflow by ID"""
        return self.user_workflows.get(workflow_id)
    
    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing user workflow"""
        try:
            if workflow_id not in self.user_workflows:
                return {"success": False, "error": "Workflow not found"}
            
            workflow = self.user_workflows[workflow_id].copy()
            
            # Apply updates
            for key, value in updates.items():
                if key in ["name", "description", "intent", "tags", "plan"]:
                    workflow[key] = value
            
            # Update metadata
            workflow["metadata"]["updated_at"] = datetime.now().isoformat()
            
            # Validate if plan was updated
            if "plan" in updates:
                if "nodes" in updates["plan"]:
                    validation_result = await self._validate_workflow_nodes(updates["plan"]["nodes"])
                    if not validation_result["valid"]:
                        return {"success": False, "error": f"Workflow validation failed: {validation_result['error']}"}
            
            self.user_workflows[workflow_id] = workflow
            
            return {"success": True, "message": "Workflow updated successfully"}
            
        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Delete a user-defined workflow"""
        try:
            if workflow_id not in self.user_workflows:
                return {"success": False, "error": "Workflow not found"}
            
            del self.user_workflows[workflow_id]
            return {"success": True, "message": "Workflow deleted successfully"}
            
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_workflow_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate workflow nodes against available agents and tools"""
        try:
            available_agents = {agent.name: agent for agent in registry.list_agents()}
            
            for node in nodes:
                # Check required fields
                required_fields = ["id", "agent", "tool"]
                for field in required_fields:
                    if field not in node:
                        return {"valid": False, "error": f"Node missing required field: {field}"}
                
                # Check agent exists
                agent_name = node["agent"]
                if agent_name not in available_agents:
                    return {"valid": False, "error": f"Agent '{agent_name}' not found"}
                
                # Check tool exists for agent
                agent = available_agents[agent_name]
                tool_name = node["tool"]
                tool_names = [tool.name for tool in agent.tools]
                if tool_name not in tool_names:
                    return {"valid": False, "error": f"Tool '{tool_name}' not found in agent '{agent_name}'"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

class EnhancedPlanner:
    """Enhanced planner with dynamic context and intelligent intent identification"""
    
    def __init__(self):
        self.workflow_manager = WorkflowManager()
        self.intent_cache = {}
        self.context_templates = {
            "agents": self._build_agents_context,
            "tools": self._build_tools_context,
            "workflows": self._build_workflows_context,
            "capabilities": self._build_capabilities_context
        }
    
    async def classify_intent_and_plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Enhanced intent classification with dynamic context and planning"""
        try:
            # Build dynamic context
            dynamic_context = await self._build_dynamic_context()
            
            # Merge with provided context
            if context:
                dynamic_context.update(context)
            
            # Classify intent using LLM with full context
            intent_result = await self._classify_intent_with_context(query, dynamic_context)
            
            if not intent_result["success"]:
                return ExecutionPlan(
                    intent="unknown",
                    plan=DAG(nodes=[]),
                    confidence=0.0,
                    reasoning=f"Intent classification failed: {intent_result['error']}"
                )
            
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
            reasoning = intent_result["reasoning"]
            suggested_agents = intent_result.get("suggested_agents", [])
            suggested_tools = intent_result.get("suggested_tools", [])
            parameters = intent_result.get("parameters", {})
            
            logger.info(f"Intent classified: {intent} (confidence: {confidence})")
            
            # Check for existing workflows that match intent
            matching_workflows = await self._find_matching_workflows(query, intent, dynamic_context)
            
            if matching_workflows:
                # Use best matching workflow
                best_workflow = matching_workflows[0]
                logger.info(f"Using workflow: {best_workflow['id']}")
                
                # Get workflow specification
                workflow_spec = None
                if best_workflow["type"] == "predefined":
                    workflow_spec = registry.get_workflow(best_workflow["id"])
                else:
                    workflow_spec = await self.workflow_manager.get_user_workflow(best_workflow["id"])
                
                if workflow_spec:
                    # Convert to DAG and apply parameter substitution
                    dag = await self._workflow_to_dag(workflow_spec, parameters, query)
                    
                    return ExecutionPlan(
                        intent=intent,
                        plan=dag,
                        confidence=confidence,
                        reasoning=f"Using workflow: {best_workflow['name']}. {reasoning}",
                        workflow_id=best_workflow["id"],
                        parameters=parameters
                    )
            
            # Generate new plan using LLM
            logger.info(f"Generating new plan for intent: {intent}")
            plan_result = await self._generate_plan_with_context(
                query, intent, dynamic_context, suggested_agents, suggested_tools, parameters
            )
            
            if plan_result["success"]:
                dag = plan_result["dag"]
                return ExecutionPlan(
                    intent=intent,
                    plan=dag,
                    confidence=confidence,
                    reasoning=f"Generated custom plan. {reasoning}",
                    parameters=parameters
                )
            else:
                return ExecutionPlan(
                    intent=intent,
                    plan=DAG(nodes=[]),
                    confidence=confidence,
                    reasoning=f"Plan generation failed: {plan_result['error']}"
                )
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return ExecutionPlan(
                intent="error",
                plan=DAG(nodes=[]),
                confidence=0.0,
                reasoning=f"Planning error: {str(e)}"
            )
    
    async def _build_dynamic_context(self) -> Dict[str, Any]:
        """Build dynamic context with current system state"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "available_agents": [],
            "available_tools": {},
            "workflow_capabilities": [],
            "system_status": "operational"
        }
        
        try:
            # Get available agents and their tools
            agents = registry.list_agents()
            context["available_agents"] = [agent.name for agent in agents]
            
            for agent in agents:
                context["available_tools"][agent.name] = [
                    {
                        "name": tool.name,
                        "description": getattr(tool, 'description', ''),
                        "params": list(tool.params_schema.get("properties", {}).keys()) if tool.params_schema else []
                    }
                    for tool in agent.tools
                ]
            
            # Get workflow capabilities
            workflows = await self.workflow_manager.list_workflows()
            context["workflow_capabilities"] = [
                {
                    "id": wf["id"],
                    "name": wf["name"],
                    "intent": wf["intent"],
                    "description": wf["description"],
                    "tags": wf.get("tags", [])
                }
                for wf in workflows
            ]
            
            # Add system capabilities summary
            context["capabilities_summary"] = {
                "total_agents": len(agents),
                "total_workflows": len(workflows),
                "main_capabilities": [
                    "web_search", "data_analysis", "summarization", 
                    "tabulation", "calculation", "sql_execution"
                ],
                "supported_intents": list(set([wf["intent"] for wf in workflows]))
            }
            
        except Exception as e:
            logger.error(f"Failed to build dynamic context: {e}")
            context["system_status"] = f"error: {str(e)}"
        
        return context
    
    async def _classify_intent_with_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify intent using LLM with full dynamic context"""
        try:
            if not llm_client or not llm_client.client:
                return self._fallback_intent_classification(query)
            
            # Build context-aware prompt
            context_summary = self._build_context_summary(context)
            
            intent_prompt = f"""
            You are an expert intent classifier for an agentic workflow system. Analyze the user query and determine the best intent and execution approach.

            SYSTEM CONTEXT:
            {context_summary}

            USER QUERY: "{query}"

            Analyze the query and determine:
            1. Primary intent (choose the most specific one)
            2. Confidence score (0-100)
            3. Suggested agents and tools to use
            4. Parameters to extract from the query
            5. Reasoning for the classification

            Available intents:
            - market_research: For researching markets, competitors, trends, analysis
            - web_search: For general web searches and information lookup
            - data_analysis: For analyzing data, extracting insights, statistical analysis
            - calculation: For mathematical calculations and computations
            - sql_query: For database queries and data retrieval
            - summarization: For text summarization and content processing
            - workflow_execution: For running complex multi-step processes
            - system_query: For system status and administrative tasks
            - custom: For unique requests requiring custom workflows

            IMPORTANT RULES:
            1. Always suggest specific agents and tools from the available context
            2. Extract any parameters mentioned in the query
            3. Consider parallel execution opportunities
            4. Prefer existing workflows when applicable
            5. Be specific about the intent - avoid generic classifications

            Return JSON response with this exact structure:
            {{
                "intent": "specific_intent_name",
                "confidence": 85,
                "reasoning": "Clear explanation of why this intent was chosen",
                "suggested_agents": ["agent1", "agent2"],
                "suggested_tools": ["tool1", "tool2"],
                "parameters": {{"key": "value"}},
                "execution_strategy": "parallel|sequential|mixed",
                "estimated_complexity": "low|medium|high"
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert intent classifier for agentic workflows. Always return valid JSON."},
                {"role": "user", "content": intent_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=800)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    # Validate response structure
                    required_fields = ["intent", "confidence", "reasoning"]
                    if all(field in result for field in required_fields):
                        return {"success": True, **result}
                    else:
                        logger.warning("LLM response missing required fields")
                        return self._fallback_intent_classification(query)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM intent response: {e}")
                    return self._fallback_intent_classification(query)
            else:
                logger.warning("No response from LLM for intent classification")
                return self._fallback_intent_classification(query)
                
        except Exception as e:
            logger.error(f"Intent classification with context failed: {e}")
            return self._fallback_intent_classification(query)
    
    def _build_context_summary(self, context: Dict[str, Any]) -> str:
        """Build a concise context summary for the LLM"""
        summary = f"""
        Available Agents: {', '.join(context.get('available_agents', []))}
        
        Agent Capabilities:
        """
        
        tools = context.get('available_tools', {})
        for agent, agent_tools in tools.items():
            tool_names = [tool['name'] for tool in agent_tools]
            summary += f"- {agent}: {', '.join(tool_names)}\n"
        
        summary += f"\nWorkflow Capabilities: {len(context.get('workflow_capabilities', []))} workflows available\n"
        summary += f"Supported Intents: {', '.join(context.get('capabilities_summary', {}).get('supported_intents', []))}\n"
        summary += f"System Status: {context.get('system_status', 'unknown')}"
        
        return summary
    
    def _fallback_intent_classification(self, query: str) -> Dict[str, Any]:
        """Fallback intent classification using rule-based approach"""
        query_lower = query.lower()
        
        # Rule-based classification
        if any(word in query_lower for word in ["market", "research", "competitor", "trend", "analysis"]):
            intent = "market_research"
            agents = ["web_search", "nlp_summarizer", "tabulator"]
            tools = ["search", "extract_insights", "ai_tabulate"]
        elif any(word in query_lower for word in ["search", "find", "lookup", "google"]):
            intent = "web_search"
            agents = ["web_search"]
            tools = ["search"]
        elif any(word in query_lower for word in ["analyze", "data", "insight", "pattern"]):
            intent = "data_analysis"
            agents = ["nlp_summarizer", "tabulator"]
            tools = ["extract_insights", "ai_tabulate"]
        elif any(word in query_lower for word in ["calculate", "compute", "math", "formula"]):
            intent = "calculation"
            agents = ["calculator"]
            tools = ["calculate"]
        elif any(word in query_lower for word in ["sql", "database", "query", "select", "table"]):
            intent = "sql_query"
            agents = ["sql_executor"]
            tools = ["execute_query"]
        elif any(word in query_lower for word in ["summarize", "summary", "brief", "overview"]):
            intent = "summarization"
            agents = ["nlp_summarizer"]
            tools = ["summarize"]
        else:
            intent = "web_search"  # Default to search
            agents = ["web_search"]
            tools = ["search"]
        
        return {
            "success": True,
            "intent": intent,
            "confidence": 65,
            "reasoning": f"Rule-based classification detected '{intent}' intent based on keywords",
            "suggested_agents": agents,
            "suggested_tools": tools,
            "parameters": {},
            "execution_strategy": "sequential"
        }
    
    async def _find_matching_workflows(self, query: str, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find existing workflows that match the query and intent"""
        try:
            workflows = context.get("workflow_capabilities", [])
            
            if not workflows:
                return []
            
            # Use LLM to find best matches if available
            if llm_client and llm_client.client:
                return await self._llm_workflow_matching(query, intent, workflows)
            else:
                return self._rule_based_workflow_matching(query, intent, workflows)
                
        except Exception as e:
            logger.error(f"Workflow matching failed: {e}")
            return []
    
    async def _llm_workflow_matching(self, query: str, intent: str, workflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to find matching workflows"""
        try:
            workflow_descriptions = []
            for wf in workflows:
                desc = f"ID: {wf['id']}, Name: {wf['name']}, Intent: {wf['intent']}, Description: {wf['description']}, Tags: {wf.get('tags', [])}"
                workflow_descriptions.append(desc)
            
            match_prompt = f"""
            Find the best matching workflows for this query and intent:
            
            Query: "{query}"
            Intent: {intent}
            
            Available workflows:
            {chr(10).join(workflow_descriptions)}
            
            Consider:
            1. Semantic similarity between query and workflow description
            2. Intent alignment
            3. Keyword matches in name and tags
            4. Functional overlap
            
            Return a JSON array of workflow IDs that match, ordered by relevance (best first):
            ["workflow_id1", "workflow_id2", ...]
            
            If no workflows match well, return an empty array: []
            """
            
            messages = [
                {"role": "system", "content": "You are a workflow matching expert. Analyze user queries and find relevant workflows."},
                {"role": "user", "content": match_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=500)
            if response:
                try:
                    matching_ids = json.loads(response.strip())
                    if isinstance(matching_ids, list):
                        # Return full workflow objects for matching IDs
                        return [wf for wf in workflows if wf["id"] in matching_ids]
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse workflow matching response: {response}")
            
            return []
            
        except Exception as e:
            logger.error(f"LLM workflow matching failed: {e}")
            return []
    
    def _rule_based_workflow_matching(self, query: str, intent: str, workflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rule-based workflow matching as fallback"""
        matches = []
        query_lower = query.lower()
        
        for workflow in workflows:
            score = 0
            
            # Intent match (high weight)
            if workflow["intent"] == intent:
                score += 50
            
            # Description keyword match
            description_lower = workflow["description"].lower()
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in description_lower:
                    score += 10
            
            # Tag match
            for tag in workflow.get("tags", []):
                if tag.lower() in query_lower:
                    score += 15
            
            # Name match
            name_lower = workflow["name"].lower()
            for word in query_words:
                if len(word) > 3 and word in name_lower:
                    score += 20
            
            if score > 30:  # Minimum threshold
                matches.append((workflow, score))
        
        # Sort by score and return workflows
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:3]]  # Return top 3
    
    async def _generate_plan_with_context(self, query: str, intent: str, context: Dict[str, Any], 
                                        suggested_agents: List[str], suggested_tools: List[str], 
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan using LLM with context"""
        try:
            if not llm_client or not llm_client.client:
                return self._fallback_plan_generation(intent, suggested_agents, suggested_tools, parameters)
            
            context_summary = self._build_context_summary(context)
            
            plan_prompt = f"""
            Generate an execution plan for this query:
            
            Query: "{query}"
            Intent: {intent}
            Suggested Agents: {suggested_agents}
            Suggested Tools: {suggested_tools}
            Parameters: {json.dumps(parameters)}
            
            SYSTEM CONTEXT:
            {context_summary}
            
            Create a DAG (Directed Acyclic Graph) with nodes that accomplish the user's request.
            
            REQUIREMENTS:
            1. Use only available agents and tools from the context
            2. Design for parallel execution where possible
            3. Ensure proper dependencies between nodes
            4. Include timeout and retry settings
            5. Use parameter substitution syntax: {{{{param_name}}}}
            
            IMPORTANT: Return ONLY a valid JSON object with this structure:
            {{
                "nodes": [
                    {{
                        "id": "unique_node_id",
                        "agent": "agent_name",
                        "tool": "tool_name", 
                        "params": {{"param": "value"}},
                        "parallel_group": "group_name_optional",
                        "depends_on": ["dependency_node_ids"],
                        "timeout_sec": 45,
                        "retries": 1
                    }}
                ]
            }}
            
            Make the plan efficient and focused on the user's actual request.
            """
            
            messages = [
                {"role": "system", "content": "You are an expert workflow planner. Generate efficient execution plans as valid JSON."},
                {"role": "user", "content": plan_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=1500)
            
            if response:
                try:
                    plan_data = json.loads(response.strip())
                    
                    if "nodes" in plan_data and isinstance(plan_data["nodes"], list):
                        # Convert to DAG
                        nodes = []
                        for node_data in plan_data["nodes"]:
                            node = Node(
                                id=node_data["id"],
                                agent=node_data["agent"],
                                tool=node_data["tool"],
                                params=node_data.get("params", {}),
                                parallel_group=node_data.get("parallel_group"),
                                depends_on=node_data.get("depends_on", []),
                                timeout_sec=node_data.get("timeout_sec", 45),
                                retries=node_data.get("retries", 1)
                            )
                            nodes.append(node)
                        
                        dag = DAG(nodes=nodes)
                        return {"success": True, "dag": dag}
                    else:
                        logger.error("Invalid plan structure from LLM")
                        return self._fallback_plan_generation(intent, suggested_agents, suggested_tools, parameters)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM plan response: {e}")
                    return self._fallback_plan_generation(intent, suggested_agents, suggested_tools, parameters)
            else:
                logger.warning("No response from LLM for plan generation")
                return self._fallback_plan_generation(intent, suggested_agents, suggested_tools, parameters)
                
        except Exception as e:
            logger.error(f"Plan generation with context failed: {e}")
            return self._fallback_plan_generation(intent, suggested_agents, suggested_tools, parameters)
    
    def _fallback_plan_generation(self, intent: str, suggested_agents: List[str], 
                                suggested_tools: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback plan generation using predefined templates"""
        try:
            nodes = []
            
            if intent == "market_research":
                nodes = [
                    Node(id="search_research", agent="web_search", tool="search", 
                         params={"query": parameters.get("topic", "market research"), "limit": 10}),
                    Node(id="summarize", agent="nlp_summarizer", tool="extract_insights",
                         params={"text": "{{results.search_research.data}}"}, depends_on=["search_research"]),
                    Node(id="tabulate", agent="tabulator", tool="ai_tabulate",
                         params={"data": "{{results.summarize.data}}"}, depends_on=["summarize"])
                ]
            elif intent == "web_search":
                nodes = [
                    Node(id="search", agent="web_search", tool="search",
                         params={"query": parameters.get("query", "search"), "limit": 10})
                ]
            elif intent == "data_analysis":
                nodes = [
                    Node(id="analyze", agent="nlp_summarizer", tool="extract_insights",
                         params={"text": parameters.get("data", ""), "insight_types": ["patterns", "trends"]}),
                    Node(id="structure", agent="tabulator", tool="ai_tabulate",
                         params={"data": "{{results.analyze.data}}"}, depends_on=["analyze"])
                ]
            elif intent == "calculation":
                nodes = [
                    Node(id="calculate", agent="calculator", tool="calculate",
                         params={"expression": parameters.get("expression", "1+1")})
                ]
            elif intent == "sql_query":
                nodes = [
                    Node(id="query", agent="sql_executor", tool="execute_query",
                         params={"query": parameters.get("query", "SELECT 1")})
                ]
            else:
                # Default to simple search
                nodes = [
                    Node(id="default_search", agent="web_search", tool="search",
                         params={"query": parameters.get("query", "search"), "limit": 5})
                ]
            
            dag = DAG(nodes=nodes)
            return {"success": True, "dag": dag}
            
        except Exception as e:
            logger.error(f"Fallback plan generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _workflow_to_dag(self, workflow_spec: Any, parameters: Dict[str, Any], query: str) -> DAG:
        """Convert workflow specification to DAG with parameter substitution"""
        try:
            if isinstance(workflow_spec, dict):
                # Handle dictionary workflow spec
                plan_data = workflow_spec.get("plan", {})
                nodes_data = plan_data.get("nodes", [])
            else:
                # Handle WorkflowSpec object
                nodes_data = workflow_spec.plan.nodes if hasattr(workflow_spec.plan, 'nodes') else []
            
            nodes = []
            
            for node_data in nodes_data:
                # Handle both dict and Node object
                if isinstance(node_data, dict):
                    node_dict = node_data
                else:
                    node_dict = node_data.dict() if hasattr(node_data, 'dict') else node_data.__dict__
                
                # Apply parameter substitution
                params = self._substitute_parameters(node_dict.get("params", {}), parameters, query)
                
                node = Node(
                    id=node_dict["id"],
                    agent=node_dict["agent"],
                    tool=node_dict["tool"],
                    params=params,
                    parallel_group=node_dict.get("parallel_group"),
                    depends_on=node_dict.get("depends_on", []),
                    timeout_sec=node_dict.get("timeout_sec", 45),
                    retries=node_dict.get("retries", 1)
                )
                nodes.append(node)
            
            return DAG(nodes=nodes)
            
        except Exception as e:
            logger.error(f"Workflow to DAG conversion failed: {e}")
            return DAG(nodes=[])
    
    def _substitute_parameters(self, params: Dict[str, Any], parameters: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Substitute parameters in workflow params"""
        substituted = {}
        
        # Add query as a default parameter
        all_params = {"query": query, **parameters}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Simple parameter substitution
                substituted_value = value
                for param_name, param_value in all_params.items():
                    placeholder = f"{{{{{param_name}}}}}"
                    if placeholder in substituted_value:
                        substituted_value = substituted_value.replace(placeholder, str(param_value))
                substituted[key] = substituted_value
            else:
                substituted[key] = value
        
        return substituted
    
    # Helper methods for building context components
    def _build_agents_context(self) -> Dict[str, Any]:
        """Build agents context section"""
        agents = registry.list_agents()
        return {
            "count": len(agents),
            "agents": [{"name": agent.name, "endpoint": agent.endpoint} for agent in agents]
        }
    
    def _build_tools_context(self) -> Dict[str, Any]:
        """Build tools context section"""
        agents = registry.list_agents()
        tools_by_agent = {}
        
        for agent in agents:
            tools_by_agent[agent.name] = [
                {"name": tool.name, "description": getattr(tool, 'description', '')}
                for tool in agent.tools
            ]
        
        return {"tools_by_agent": tools_by_agent}
    
    def _build_workflows_context(self) -> Dict[str, Any]:
        """Build workflows context section"""
        workflow_ids = registry.list_workflows()
        workflows = []
        
        for workflow_id in workflow_ids:
            try:
                spec = registry.get_workflow(workflow_id)
                if spec:
                    workflows.append({
                        "id": spec.id,
                        "name": spec.name,
                        "intent": spec.intent,
                        "description": spec.description
                    })
            except Exception as e:
                logger.warning(f"Failed to load workflow {workflow_id}: {e}")
        
        return {"count": len(workflows), "workflows": workflows}
    
    def _build_capabilities_context(self) -> Dict[str, Any]:
        """Build system capabilities context"""
        return {
            "core_capabilities": [
                "web_search", "data_analysis", "summarization", 
                "tabulation", "calculation", "sql_execution"
            ],
            "execution_modes": ["sequential", "parallel", "mixed"],
            "output_formats": ["text", "table", "json", "report"]
        }

# Create global instance
planner = EnhancedPlanner()