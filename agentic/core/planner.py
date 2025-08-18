# agentic/core/planner.py
import json
import re
from typing import Dict, Any, Optional, List
import logging
from .types import DAG, Node, QueryRequest, WorkflowSpec
from .registry import registry
from .llm_client import llm_client

logger = logging.getLogger(__name__)

class SystemQueryHandler:
    """Handle system-level queries that don't require agent orchestration"""
    
    def __init__(self):
        self.system_patterns = {
            "list_tools": [
                "list tools", "available tools", "what tools", "show tools",
                "list available tools", "what tools are available", "show me tools"
            ],
            "list_agents": [
                "list agents", "available agents", "what agents", "show agents",
                "list available agents", "what agents are available", "show me agents"
            ],
            "list_workflows": [
                "list workflows", "available workflows", "what workflows", "show workflows",
                "list available workflows", "what workflows are available"
            ],
            "system_status": [
                "system status", "health check", "status", "how are you",
                "are you working", "system health"
            ],
            "help": [
                "help", "what can you do", "how to use", "instructions",
                "what are your capabilities", "usage"
            ]
        }
    
    def is_system_query(self, query: str) -> Optional[str]:
        """Check if query is a system query and return the type"""
        query_lower = query.lower().strip()
        
        for query_type, patterns in self.system_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return query_type
        
        return None
    
    def handle_system_query(self, query_type: str, query: str) -> DAG:
        """Create a simple DAG that returns system information"""
        
        if query_type == "list_tools":
            # Create a simple node that lists all tools
            node = Node(
                id="list_all_tools",
                agent="system",
                tool="list_tools",
                params={"include_details": True},
                description="List all available tools across agents"
            )
        elif query_type == "list_agents":
            node = Node(
                id="list_all_agents",
                agent="system", 
                tool="list_agents",
                params={"include_health": True},
                description="List all available agents"
            )
        elif query_type == "list_workflows":
            node = Node(
                id="list_all_workflows",
                agent="system",
                tool="list_workflows", 
                params={},
                description="List all available workflows"
            )
        elif query_type == "system_status":
            node = Node(
                id="system_status",
                agent="system",
                tool="system_status",
                params={},
                description="Get system health and status"
            )
        else:  # help
            node = Node(
                id="show_help",
                agent="system",
                tool="help",
                params={"query": query},
                description="Show help and capabilities"
            )
        
        return DAG(
            nodes=[node],
            metadata={
                "intent": "system_query",
                "query": query,
                "system_query_type": query_type,
                "generated": True
            }
        )

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            "market_research": [
                "market research", "competitive analysis", "industry analysis", 
                "market trends", "competitor", "market share", "industry report"
            ],
            "data_analysis": [
                "analyze data", "data analysis", "summarize data", "tabulate",
                "create table", "process data", "data insights"
            ],
            "web_search": [
                "search", "find information", "lookup", "research", 
                "what is", "tell me about", "information about"
            ],
            "calculation": [
                "calculate", "compute", "math", "arithmetic", "add", "multiply",
                "divide", "subtract", "percentage", "interest"
            ],
            "sql_query": ["sql", "database", "query", "select", "table", "records"]
        }
        self.system_handler = SystemQueryHandler()
    
    async def classify(self, query: str) -> str:
        """Classify user intent from query using both rules and LLM"""
        query_lower = query.lower()
        
        # Check for system queries first
        system_query_type = self.system_handler.is_system_query(query)
        if system_query_type:
            return "system_query"
        
        # Rule-based classification first (fast)
        rule_based_intent = self._rule_based_classify(query_lower)
        
        # Use LLM for more nuanced classification
        if llm_client and llm_client.client:
            try:
                llm_intent = await llm_client.classify_intent(query)
                if llm_intent and llm_intent != "general":
                    logger.info(f"LLM classified intent: {llm_intent} (rule-based: {rule_based_intent})")
                    return llm_intent
            except Exception as e:
                logger.error(f"LLM intent classification failed: {e}")
        
        logger.info(f"Using rule-based intent: {rule_based_intent}")
        return rule_based_intent
    
    def _rule_based_classify(self, query_lower: str) -> str:
        """Rule-based intent classification"""
        # Check for exact matches first
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent
        
        # More specific pattern matching
        if any(word in query_lower for word in ["market", "industry", "competitive", "competitor"]):
            return "market_research"
        
        if any(word in query_lower for word in ["search", "find", "lookup", "information"]):
            return "web_search"
        
        if any(word in query_lower for word in ["analyze", "table", "data", "summarize"]):
            return "data_analysis"
        
        if any(word in query_lower for word in ["calculate", "compute", "+", "-", "*", "/"]):
            return "calculation"
        
        return "general"

class PlanGenerator:
    def __init__(self):
        self.system_handler = SystemQueryHandler()
    
    async def generate_dynamic_plan(self, intent: str, query: str) -> DAG:
        """Generate execution plan using LLM"""
        
        # Handle system queries first
        if intent == "system_query":
            system_query_type = self.system_handler.is_system_query(query)
            if system_query_type:
                return self.system_handler.handle_system_query(system_query_type, query)
        
        if not llm_client or not llm_client.client:
            return self._fallback_plan_generation(intent, query)
        
        try:
            # Get available agents and tools
            agents = registry.list_agents()
            agent_info = []
            for agent in agents:
                if agent.enabled:
                    tools = [f"{tool.name}: {tool.description}" for tool in agent.tools]
                    agent_info.append(f"{agent.name} - {', '.join(tools)}")
            
            plan_prompt = f"""
            Generate a workflow plan for this query: "{query}"
            Intent: {intent}
            
            Available agents and tools:
            {chr(10).join(agent_info)}
            
            Create a JSON workflow plan with these rules:
            1. Use parallel_group for independent operations
            2. Use depends_on for sequential dependencies
            3. Use descriptive node IDs
            4. Parameters can reference previous results with {{{{results.node_id.data}}}}
            5. For parallel groups, use {{{{results.group_name.combined}}}}
            6. NEVER use agent "none" - always specify a valid agent name from the list above
            7. Every node must have a valid agent that exists in the available agents list
            
            Return only valid JSON in this format:
            {{
              "nodes": [
                {{
                  "id": "step_name",
                  "agent": "agent_name",
                  "tool": "tool_name",
                  "params": {{"param": "value"}},
                  "parallel_group": "group_name",
                  "depends_on": ["dependency"],
                  "description": "What this step does"
                }}
              ]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a workflow planner. Generate efficient execution plans using only the available agents and tools provided."},
                {"role": "user", "content": plan_prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.1)
            if response:
                try:
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        plan_data = json.loads(json_match.group())
                        nodes_data = plan_data.get("nodes", [])
                        
                        # Validate that all nodes have valid agents
                        valid_nodes = []
                        available_agent_names = {agent.name for agent in agents if agent.enabled}
                        
                        for node_data in nodes_data:
                            agent_name = node_data.get("agent")
                            if agent_name in available_agent_names:
                                valid_nodes.append(Node(**node_data))
                            else:
                                logger.warning(f"Skipping node with invalid agent: {agent_name}")
                        
                        if valid_nodes:
                            logger.info(f"Generated dynamic plan with {len(valid_nodes)} valid nodes")
                            return DAG(nodes=valid_nodes, metadata={"intent": intent, "query": query, "generated": True})
                        else:
                            logger.warning("No valid nodes in generated plan, falling back to template")
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM-generated plan: {e}")
            
        except Exception as e:
            logger.error(f"Dynamic plan generation failed: {e}")
        
        # Fallback to template-based generation
        return self._fallback_plan_generation(intent, query)
    
    def _fallback_plan_generation(self, intent: str, query: str) -> DAG:
        """Fallback plan generation using templates"""
        if intent == "market_research":
            return self._create_market_research_plan(query)
        elif intent == "web_search":
            return self._create_search_plan(query)
        elif intent == "data_analysis":
            return self._create_analysis_plan(query)
        elif intent == "calculation":
            return self._create_calculation_plan(query)
        elif intent == "system_query":
            # Fallback system query handling
            system_query_type = self.system_handler.is_system_query(query)
            if system_query_type:
                return self.system_handler.handle_system_query(system_query_type, query)
            else:
                return self._create_default_plan(query)
        else:
            return self._create_default_plan(query)
    
    def _create_market_research_plan(self, query: str) -> DAG:
        """Create market research workflow"""
        # Extract topic from query
        topic = query.replace("market research", "").replace("research", "").strip()
        if not topic:
            topic = "market analysis"
        
        nodes = [
            Node(
                id="search_trends",
                agent="web_search",
                tool="search",
                params={"query": f"{topic} market trends 2024 2025", "limit": 8},
                parallel_group="market_research",
                description=f"Search for {topic} market trends"
            ),
            Node(
                id="search_competitors", 
                agent="web_search",
                tool="search",
                params={"query": f"{topic} competitors competitive analysis", "limit": 8},
                parallel_group="market_research",
                description=f"Search for {topic} competitive landscape"
            ),
            Node(
                id="search_opportunities",
                agent="web_search",
                tool="search",
                params={"query": f"{topic} market opportunities challenges", "limit": 6},
                parallel_group="market_research",
                description=f"Search for {topic} market opportunities"
            ),
            Node(
                id="summarize_findings",
                agent="nlp_summarizer",
                tool="summarize",
                params={
                    "text": "{{results.market_research.combined}}",
                    "style": "detailed",
                    "max_length": 400
                },
                depends_on=["market_research"],
                description="Summarize market research findings"
            ),
            Node(
                id="create_report",
                agent="tabulator", 
                tool="tabulate",
                params={
                    "data": "{{results.summarize_findings.data}}",
                    "fields": ["topic", "key_finding", "source", "relevance"],
                    "format": "json"
                },
                depends_on=["summarize_findings"],
                description="Create structured market research report"
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "market_research", "query": query, "topic": topic})
    
    def _create_search_plan(self, query: str) -> DAG:
        """Create simple search workflow"""
        nodes = [
            Node(
                id="web_search",
                agent="web_search", 
                tool="search",
                params={"query": query, "limit": 10},
                description=f"Search for information about: {query}"
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "web_search", "query": query})
    
    def _create_analysis_plan(self, query: str) -> DAG:
        """Create data analysis workflow"""
        nodes = [
            Node(
                id="analyze_content",
                agent="nlp_summarizer",
                tool="summarize",
                params={
                    "text": query,  # Assume query contains data to analyze
                    "style": "detailed",
                    "max_length": 300
                },
                description="Analyze the provided content"
            ),
            Node(
                id="structure_results",
                agent="tabulator",
                tool="tabulate", 
                params={
                    "data": "{{results.analyze_content.data}}",
                    "format": "json"
                },
                depends_on=["analyze_content"],
                description="Structure analysis results"
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "data_analysis", "query": query})
    
    def _create_calculation_plan(self, query: str) -> DAG:
        """Create calculation workflow"""
        # Try to extract numbers and operation from query
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            operation = "add"
            
            if any(op in query.lower() for op in ["multiply", "*", "times"]):
                operation = "multiply"
            elif any(op in query.lower() for op in ["divide", "/", "divided"]):
                operation = "divide"
            elif any(op in query.lower() for op in ["subtract", "-", "minus"]):
                operation = "subtract"
            elif any(op in query.lower() for op in ["power", "^", "**", "raised"]):
                operation = "power"
            
            nodes = [
                Node(
                    id="calculate",
                    agent="calculator",
                    tool=operation,
                    params={"a": a, "b": b},
                    description=f"Calculate {a} {operation} {b}"
                )
            ]
        else:
            # Default to simple addition
            nodes = [
                Node(
                    id="calculate",
                    agent="calculator",
                    tool="add",
                    params={"a": 0, "b": 0},
                    description="Default calculation"
                )
            ]
        
        return DAG(nodes=nodes, metadata={"intent": "calculation", "query": query})
    
    def _create_default_plan(self, query: str) -> DAG:
        """Create default workflow for unknown intents"""
        nodes = [
            Node(
                id="general_search",
                agent="web_search",
                tool="search",
                params={"query": query, "limit": 5},
                description=f"General search for: {query}"
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "general", "query": query})

class Planner:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.generator = PlanGenerator()
    
    async def create_plan(self, request: QueryRequest) -> DAG:
        """Create execution plan from user request"""
        logger.info(f"Creating plan for query: {request.text}")
        
        # Check for predefined workflow first
        workflow_id = request.options.get("use_workflow")
        if workflow_id:
            workflow = registry.get_workflow(workflow_id)
            if workflow:
                logger.info(f"Using predefined workflow: {workflow_id}")
                return workflow.plan
        
        # Classify intent
        intent = await self.classifier.classify(request.text)
        logger.info(f"Classified intent: {intent}")
        
        # Handle system queries
        if intent == "system_query":
            return await self.generator.generate_dynamic_plan(intent, request.text)
        
        # Check for intent-based workflow
        workflows = registry.list_workflows()
        for workflow in workflows:
            if workflow.intent == intent:
                logger.info(f"Using intent-based workflow: {workflow.id}")
                return workflow.plan
        
        # Generate plan dynamically
        logger.info("Generating dynamic plan")
        return await self.generator.generate_dynamic_plan(intent, request.text)

planner = Planner()