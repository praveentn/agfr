# tests/test_planner.py
import pytest
from unittest.mock import AsyncMock, patch

from agentic.core.planner import IntentClassifier, Planner
from agentic.core.types import QueryRequest


class TestIntentClassifier:
    
    @pytest.mark.asyncio
    async def test_classify_market_research(self):
        """Test classifying market research intent"""
        classifier = IntentClassifier()
        
        intent = await classifier.classify("Do market research on running shoes")
        assert intent == "market_research"
        
        intent = await classifier.classify("Analyze the competitive landscape")
        assert intent == "market_research"
    
    @pytest.mark.asyncio
    async def test_classify_web_search(self):
        """Test classifying web search intent"""
        classifier = IntentClassifier()
        
        intent = await classifier.classify("Search for Python tutorials")
        assert intent == "web_search"
        
        intent = await classifier.classify("Find information about FastAPI")
        assert intent == "web_search"
    
    @pytest.mark.asyncio
    async def test_classify_calculation(self):
        """Test classifying calculation intent"""
        classifier = IntentClassifier()
        
        intent = await classifier.classify("Calculate 15% of 1000")
        assert intent == "calculation"
        
        intent = await classifier.classify("Compute the compound interest")
        assert intent == "calculation"
    
    @pytest.mark.asyncio
    async def test_classify_general(self):
        """Test classifying general intent"""
        classifier = IntentClassifier()
        
        intent = await classifier.classify("Hello, how are you?")
        assert intent == "general"


class TestPlanner:
    
    @pytest.mark.asyncio
    async def test_create_plan_with_workflow(self, mock_registry):
        """Test creating plan with specified workflow"""
        planner = Planner()
        planner.classifier = AsyncMock()
        
        # Mock registry to return test workflow
        with patch('agentic.core.planner.registry', mock_registry):
            request = QueryRequest(
                text="Test query",
                options={"use_workflow": "test_workflow"}
            )
            
            dag = await planner.create_plan(request)
            assert len(dag.nodes) == 1
            assert dag.nodes[0].agent == "test_agent"
            assert dag.nodes[0].tool == "test_tool"
    
    @pytest.mark.asyncio
    async def test_create_plan_market_research(self):
        """Test creating market research plan"""
        planner = Planner()
        planner.classifier.classify = AsyncMock(return_value="market_research")
        
        request = QueryRequest(text="Research the shoe market")
        dag = await planner.create_plan(request)
        
        # Should create market research workflow
        assert len(dag.nodes) >= 2  # At least search and summary steps
        assert dag.metadata["intent"] == "market_research"
    
    @pytest.mark.asyncio
    async def test_create_plan_web_search(self):
        """Test creating web search plan"""
        planner = Planner()
        planner.classifier.classify = AsyncMock(return_value="web_search")
        
        request = QueryRequest(text="Search for Python tutorials")
        dag = await planner.create_plan(request)
        
        # Should create simple search workflow
        assert len(dag.nodes) == 1
        assert dag.nodes[0].agent == "web_search"
        assert dag.nodes[0].tool == "search"
        assert dag.metadata["intent"] == "web_search"

