# ============================================================================
# agentic/core/planner.py
import json
import re
from typing import Dict, Any, Optional, List
import logging
from .types import DAG, Node, QueryRequest, WorkflowSpec
from .registry import registry
from .llm_client import llm_client

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            "market_research": ["market research", "competitive analysis", "industry analysis", "market trends"],
            "data_analysis": ["analyze data", "data analysis", "summarize data", "tabulate"],
            "web_search": ["search", "find information", "lookup", "research"],
            "calculation": ["calculate", "compute", "math", "arithmetic"],
            "sql_query": ["sql", "database", "query", "select"]
        }
    
    async def classify(self, query: str) -> str:
        """Classify user intent from query"""
        query_lower = query.lower()
        
        # Rule-based classification first
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        # LLM-based classification as fallback
        prompt = registry.get_prompt("intent/classifier")
        if prompt and llm_client:
            try:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Classify this query: {query}"}
                ]
                response = await llm_client.generate(messages)
                if response and any(intent in response.lower() for intent in self.intent_patterns.keys()):
                    for intent in self.intent_patterns.keys():
                        if intent in response.lower():
                            return intent
            except Exception as e:
                logger.error(f"LLM intent classification failed: {e}")
        
        return "general"

class Planner:
    def __init__(self):
        self.classifier = IntentClassifier()
    
    async def create_plan(self, request: QueryRequest) -> DAG:
        """Create execution plan from user request"""
        intent = await self.classifier.classify(request.text)
        
        # Check for predefined workflow
        workflow_id = request.options.get("use_workflow")
        if workflow_id:
            workflow = registry.get_workflow(workflow_id)
            if workflow:
                return workflow.plan
        
        # Check for intent-based workflow
        workflows = registry.list_workflows()
        for workflow in workflows:
            if workflow.intent == intent:
                return workflow.plan
        
        # Generate plan dynamically
        return await self._generate_plan(intent, request.text)
    
    async def _generate_plan(self, intent: str, query: str) -> DAG:
        """Generate execution plan using LLM"""
        if intent == "market_research":
            return self._create_market_research_plan(query)
        elif intent == "web_search":
            return self._create_search_plan(query)
        elif intent == "data_analysis":
            return self._create_analysis_plan(query)
        else:
            return self._create_default_plan(query)
    
    def _create_market_research_plan(self, query: str) -> DAG:
        """Create market research workflow"""
        nodes = [
            Node(
                id="search1",
                agent="web_search",
                tool="search",
                params={"query": f"{query} market trends 2024", "limit": 10},
                parallel_group="searches"
            ),
            Node(
                id="search2", 
                agent="web_search",
                tool="search",
                params={"query": f"{query} competitive analysis", "limit": 10},
                parallel_group="searches"
            ),
            Node(
                id="summarize",
                agent="nlp_summarizer",
                tool="summarize",
                params={"text": "{{results.searches.combined}}"},
                depends_on=["searches"]
            ),
            Node(
                id="tabulate",
                agent="tabulator", 
                tool="tabulate",
                params={"data": "{{results.summarize.data}}", "fields": ["topic", "findings", "sources"]},
                depends_on=["summarize"]
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "market_research", "query": query})
    
    def _create_search_plan(self, query: str) -> DAG:
        """Create simple search workflow"""
        nodes = [
            Node(
                id="search",
                agent="web_search", 
                tool="search",
                params={"query": query, "limit": 10}
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "web_search", "query": query})
    
    def _create_analysis_plan(self, query: str) -> DAG:
        """Create data analysis workflow"""
        nodes = [
            Node(
                id="analyze",
                agent="data_analyzer",
                tool="analyze",
                params={"query": query}
            ),
            Node(
                id="tabulate",
                agent="tabulator",
                tool="tabulate", 
                params={"data": "{{results.analyze.data}}"},
                depends_on=["analyze"]
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "data_analysis", "query": query})
    
    def _create_default_plan(self, query: str) -> DAG:
        """Create default workflow for unknown intents"""
        nodes = [
            Node(
                id="search",
                agent="web_search",
                tool="search",
                params={"query": query, "limit": 5}
            )
        ]
        return DAG(nodes=nodes, metadata={"intent": "general", "query": query})

planner = Planner()

