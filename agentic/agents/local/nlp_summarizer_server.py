# agentic/agents/local/nlp_summarizer_server.py
import json
import time
import uuid
import re
import sys
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agentic.core.llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNLPSummarizerMCPServer:
    """Enhanced NLP Summarizer with full Azure OpenAI integration and intelligent fallbacks"""
    
    def __init__(self):
        self.app = FastAPI(title="Enhanced NLP Summarizer MCP Server")
        self.sessions = {}
        self.tools = [
            {
                "name": "summarize",
                "description": "Intelligent text summarization using Azure OpenAI with multiple styles and formats",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize", "minLength": 10},
                        "max_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300, "description": "Maximum summary length in words"},
                        "style": {"type": "string", "enum": ["brief", "detailed", "bullet_points", "executive", "technical"], "default": "brief", "description": "Summary style and format"},
                        "focus": {"type": "string", "description": "Specific aspect to focus on (optional)", "examples": ["key findings", "recommendations", "financial data"]},
                        "tone": {"type": "string", "enum": ["professional", "casual", "academic", "persuasive"], "default": "professional", "description": "Tone of the summary"},
                        "include_citations": {"type": "boolean", "default": False, "description": "Include source references if available"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "extract_entities",
                "description": "Extract named entities and key information using Azure OpenAI with intelligent parsing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze for entities", "minLength": 5},
                        "entity_types": {"type": "array", "items": {"type": "string"}, "description": "Types of entities to extract", "examples": ["person", "organization", "location", "date", "money", "product"]},
                        "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "Minimum confidence threshold for extraction"},
                        "include_context": {"type": "boolean", "default": True, "description": "Include surrounding context for entities"},
                        "output_format": {"type": "string", "enum": ["structured", "list", "highlighted"], "default": "structured", "description": "Output format preference"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "analyze_sentiment",
                "description": "Analyze sentiment and emotional tone using Azure OpenAI with detailed insights",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze for sentiment", "minLength": 5},
                        "granularity": {"type": "string", "enum": ["document", "sentence", "aspect"], "default": "document", "description": "Level of sentiment analysis"},
                        "aspects": {"type": "array", "items": {"type": "string"}, "description": "Specific aspects to analyze sentiment for"},
                        "include_emotions": {"type": "boolean", "default": True, "description": "Include detailed emotion analysis"},
                        "confidence_scores": {"type": "boolean", "default": True, "description": "Include confidence scores"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "extract_insights",
                "description": "Extract business insights and actionable information using Azure OpenAI",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze for insights", "minLength": 20},
                        "insight_types": {"type": "array", "items": {"type": "string"}, "description": "Types of insights to extract", "examples": ["trends", "opportunities", "risks", "recommendations", "key_metrics"]},
                        "business_context": {"type": "string", "description": "Business context for insight extraction", "examples": ["market_research", "customer_feedback", "financial_analysis"]},
                        "priority": {"type": "string", "enum": ["high_level", "detailed", "comprehensive"], "default": "detailed", "description": "Depth of insight analysis"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "compare_texts",
                "description": "Compare multiple texts and identify similarities, differences, and relationships",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "texts": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5, "description": "List of texts to compare"},
                        "comparison_type": {"type": "string", "enum": ["similarity", "differences", "evolution", "sentiment_shift"], "default": "similarity", "description": "Type of comparison to perform"},
                        "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Specific areas to focus comparison on"},
                        "output_format": {"type": "string", "enum": ["narrative", "table", "bullet_points"], "default": "narrative", "description": "Format of comparison output"}
                    },
                    "required": ["texts"]
                }
            },
            {
                "name": "analyze_query",
                "description": "Analyze user query intent and provide structured response with recommendations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User query to analyze", "minLength": 3},
                        "context": {"type": "string", "description": "Additional context for analysis"},
                        "response_type": {"type": "string", "enum": ["intent_only", "full_analysis", "recommendations"], "default": "full_analysis", "description": "Type of analysis to perform"},
                        "include_suggestions": {"type": "boolean", "default": True, "description": "Include query improvement suggestions"}
                    },
                    "required": ["query"]
                }
            }
        ]
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/mcp/info")
        async def get_server_info():
            """Get server information"""
            return {
                "name": "Enhanced NLP Summarizer Server",
                "version": "2.0.0",
                "description": "Advanced NLP processing with Azure OpenAI integration",
                "capabilities": [
                    "text_summarization", "entity_extraction", "sentiment_analysis",
                    "insight_extraction", "text_comparison", "query_analysis"
                ],
                "protocol": "mcp-2025-06-18",
                "llm_integration": "azure_openai" if llm_client and llm_client.client else "fallback"
            }
        
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List available tools"""
            return {"tools": self.tools}
        
        @self.app.post("/mcp/call")
        async def call_tool(request: Request):
            """Call a specific tool"""
            try:
                data = await request.json()
                tool_name = data.get("name")
                arguments = data.get("arguments", {})
                
                if tool_name not in [tool["name"] for tool in self.tools]:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                # Route to appropriate handler
                if tool_name == "summarize":
                    result = await self._summarize(arguments)
                elif tool_name == "extract_entities":
                    result = await self._extract_entities(arguments)
                elif tool_name == "analyze_sentiment":
                    result = await self._analyze_sentiment(arguments)
                elif tool_name == "extract_insights":
                    result = await self._extract_insights(arguments)
                elif tool_name == "compare_texts":
                    result = await self._compare_texts(arguments)
                elif tool_name == "analyze_query":
                    result = await self._analyze_query(arguments)
                else:
                    raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not implemented")
                
                return {"success": True, "result": result}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Test Azure OpenAI connection
                llm_status = "healthy" if llm_client and llm_client.client else "not_configured"
                
                return {
                    "status": "healthy",
                    "server": "Enhanced NLP Summarizer Server",
                    "azure_openai": llm_status,
                    "timestamp": time.time(),
                    "capabilities": [
                        "summarize", "extract_entities", "analyze_sentiment",
                        "extract_insights", "compare_texts", "analyze_query"
                    ]
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        @self.app.get("/events")
        async def event_stream(session_id: Optional[str] = None):
            """Server-sent events stream"""
            return StreamingResponse(
                self._sse_stream(session_id), 
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
    
    def _setup_middleware(self):
        """Setup middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Core NLP Tools with Azure OpenAI Integration
    
    async def _summarize(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced text summarization with Azure OpenAI"""
        try:
            text = arguments.get("text", "").strip()
            max_length = arguments.get("max_length", 300)
            style = arguments.get("style", "brief")
            focus = arguments.get("focus", "")
            tone = arguments.get("tone", "professional")
            include_citations = arguments.get("include_citations", False)
            
            if not text or len(text) < 10:
                return {"success": False, "error": "Text must be at least 10 characters long"}
            
            # Use Azure OpenAI if available
            if llm_client and llm_client.client:
                return await self._ai_summarize(text, max_length, style, focus, tone, include_citations)
            else:
                return await self._fallback_summarize(text, max_length, style)
                
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_summarize(self, text: str, max_length: int, style: str, focus: str, tone: str, include_citations: bool) -> Dict[str, Any]:
        """AI-powered summarization using Azure OpenAI"""
        try:
            # Build context-aware prompt
            style_instructions = {
                "brief": "Create a concise summary focusing on the most essential points",
                "detailed": "Provide a comprehensive summary with key details and context",
                "bullet_points": "Structure the summary as clear bullet points",
                "executive": "Write an executive summary suitable for business leaders",
                "technical": "Focus on technical details and specifications"
            }
            
            focus_instruction = f" Pay special attention to: {focus}." if focus else ""
            citation_instruction = " Include source references where applicable." if include_citations else ""
            
            prompt = f"""
            Summarize the following text in a {tone} tone using a {style} style.
            Maximum length: approximately {max_length} words.
            {style_instructions.get(style, '')}{focus_instruction}{citation_instruction}
            
            Text to summarize:
            {text}
            
            Provide the summary along with:
            1. Key points (3-5 main takeaways)
            2. Word count of the summary
            3. Confidence score (0-100) for summary quality
            4. Any important details that were omitted due to length constraints
            
            Format as JSON:
            {{
                "summary": "...",
                "key_points": ["point1", "point2", "point3"],
                "word_count": 150,
                "confidence": 95,
                "omitted_details": ["detail1", "detail2"],
                "focus_areas": ["area1", "area2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert text summarizer. Always provide accurate, concise summaries in the requested format."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=800)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    # Validate response structure
                    if "summary" in result:
                        return {
                            "success": True,
                            "summary": result["summary"],
                            "key_points": result.get("key_points", []),
                            "word_count": result.get("word_count", len(result["summary"].split())),
                            "confidence": result.get("confidence", 85),
                            "style": style,
                            "tone": tone,
                            "omitted_details": result.get("omitted_details", []),
                            "focus_areas": result.get("focus_areas", []),
                            "method": "azure_openai"
                        }
                    else:
                        # Fallback if JSON parsing fails but we have content
                        return await self._fallback_summarize(text, max_length, style)
                        
                except json.JSONDecodeError:
                    # Use response as summary if JSON parsing fails
                    summary_text = response.strip()
                    return {
                        "success": True,
                        "summary": summary_text,
                        "key_points": self._extract_key_points_from_text(summary_text),
                        "word_count": len(summary_text.split()),
                        "confidence": 75,
                        "style": style,
                        "method": "azure_openai_text"
                    }
            else:
                return await self._fallback_summarize(text, max_length, style)
                
        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            return await self._fallback_summarize(text, max_length, style)
    
    async def _extract_entities(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced entity extraction with Azure OpenAI"""
        try:
            text = arguments.get("text", "").strip()
            entity_types = arguments.get("entity_types", ["person", "organization", "location", "date"])
            confidence_threshold = arguments.get("confidence_threshold", 0.7)
            include_context = arguments.get("include_context", True)
            output_format = arguments.get("output_format", "structured")
            
            if not text:
                return {"success": False, "error": "Text is required for entity extraction"}
            
            # Use Azure OpenAI if available
            if llm_client and llm_client.client:
                return await self._ai_extract_entities(text, entity_types, confidence_threshold, include_context, output_format)
            else:
                return await self._fallback_extract_entities(text, entity_types)
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_extract_entities(self, text: str, entity_types: List[str], confidence_threshold: float, include_context: bool, output_format: str) -> Dict[str, Any]:
        """AI-powered entity extraction using Azure OpenAI"""
        try:
            entity_list = ", ".join(entity_types)
            context_instruction = " Include surrounding context (2-3 words before/after)." if include_context else ""
            
            prompt = f"""
            Extract named entities from the following text. 
            Entity types to find: {entity_list}
            Confidence threshold: {confidence_threshold}
            {context_instruction}
            
            Text: {text}
            
            For each entity found, provide:
            1. Entity text
            2. Entity type
            3. Confidence score (0.0 to 1.0)
            4. Position in text (start, end)
            5. Context (if requested)
            
            Return as JSON:
            {{
                "entities": [
                    {{
                        "text": "entity_text",
                        "type": "entity_type",
                        "confidence": 0.95,
                        "start_pos": 10,
                        "end_pos": 20,
                        "context": "surrounding words"
                    }}
                ],
                "total_found": 5,
                "entity_counts": {{"person": 2, "organization": 3}}
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert named entity recognition system. Extract entities accurately with confidence scores."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.1, max_tokens=1000)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    if "entities" in result:
                        # Filter by confidence threshold
                        filtered_entities = [
                            entity for entity in result["entities"]
                            if entity.get("confidence", 0) >= confidence_threshold
                        ]
                        
                        # Format according to output_format
                        formatted_result = self._format_entities(filtered_entities, output_format)
                        
                        return {
                            "success": True,
                            "entities": filtered_entities,
                            "formatted_output": formatted_result,
                            "total_found": len(filtered_entities),
                            "entity_counts": self._count_entity_types(filtered_entities),
                            "confidence_threshold": confidence_threshold,
                            "method": "azure_openai"
                        }
                    else:
                        return await self._fallback_extract_entities(text, entity_types)
                        
                except json.JSONDecodeError:
                    return await self._fallback_extract_entities(text, entity_types)
            else:
                return await self._fallback_extract_entities(text, entity_types)
                
        except Exception as e:
            logger.error(f"AI entity extraction failed: {e}")
            return await self._fallback_extract_entities(text, entity_types)
    
    async def _analyze_sentiment(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced sentiment analysis with Azure OpenAI"""
        try:
            text = arguments.get("text", "").strip()
            granularity = arguments.get("granularity", "document")
            aspects = arguments.get("aspects", [])
            include_emotions = arguments.get("include_emotions", True)
            confidence_scores = arguments.get("confidence_scores", True)
            
            if not text:
                return {"success": False, "error": "Text is required for sentiment analysis"}
            
            # Use Azure OpenAI if available
            if llm_client and llm_client.client:
                return await self._ai_analyze_sentiment(text, granularity, aspects, include_emotions, confidence_scores)
            else:
                return await self._fallback_analyze_sentiment(text)
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_analyze_sentiment(self, text: str, granularity: str, aspects: List[str], include_emotions: bool, confidence_scores: bool) -> Dict[str, Any]:
        """AI-powered sentiment analysis using Azure OpenAI"""
        try:
            emotion_instruction = " Also analyze emotions (joy, anger, fear, sadness, surprise, disgust)." if include_emotions else ""
            confidence_instruction = " Include confidence scores for all analyses." if confidence_scores else ""
            aspect_instruction = f" Analyze sentiment for these specific aspects: {', '.join(aspects)}." if aspects else ""
            
            granularity_instructions = {
                "document": "Analyze the overall sentiment of the entire text.",
                "sentence": "Analyze sentiment for each sentence separately.",
                "aspect": "Focus on aspect-based sentiment analysis."
            }
            
            prompt = f"""
            Perform sentiment analysis on the following text.
            Analysis level: {granularity_instructions.get(granularity, "document level")}
            {emotion_instruction}{confidence_instruction}{aspect_instruction}
            
            Text: {text}
            
            Provide analysis in JSON format:
            {{
                "overall_sentiment": "positive/negative/neutral",
                "overall_confidence": 0.95,
                "sentiment_score": 0.8,
                "emotions": {{
                    "joy": 0.6,
                    "anger": 0.1,
                    "fear": 0.0,
                    "sadness": 0.2,
                    "surprise": 0.1,
                    "disgust": 0.0
                }},
                "key_phrases": ["positive phrase", "negative phrase"],
                "reasoning": "Brief explanation of sentiment classification",
                "aspects": {{
                    "aspect1": {{"sentiment": "positive", "confidence": 0.9}},
                    "aspect2": {{"sentiment": "negative", "confidence": 0.8}}
                }}
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert sentiment analysis system. Provide detailed, accurate sentiment analysis with confidence scores."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=800)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    return {
                        "success": True,
                        "overall_sentiment": result.get("overall_sentiment", "neutral"),
                        "confidence": result.get("overall_confidence", 0.75),
                        "sentiment_score": result.get("sentiment_score", 0.0),
                        "emotions": result.get("emotions", {}),
                        "key_phrases": result.get("key_phrases", []),
                        "reasoning": result.get("reasoning", ""),
                        "aspects": result.get("aspects", {}),
                        "granularity": granularity,
                        "method": "azure_openai"
                    }
                    
                except json.JSONDecodeError:
                    return await self._fallback_analyze_sentiment(text)
            else:
                return await self._fallback_analyze_sentiment(text)
                
        except Exception as e:
            logger.error(f"AI sentiment analysis failed: {e}")
            return await self._fallback_analyze_sentiment(text)
    
    async def _extract_insights(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business insights using Azure OpenAI"""
        try:
            text = arguments.get("text", "").strip()
            insight_types = arguments.get("insight_types", ["trends", "opportunities", "recommendations"])
            business_context = arguments.get("business_context", "general")
            priority = arguments.get("priority", "detailed")
            
            if not text:
                return {"success": False, "error": "Text is required for insight extraction"}
            
            # Use Azure OpenAI if available
            if llm_client and llm_client.client:
                return await self._ai_extract_insights(text, insight_types, business_context, priority)
            else:
                return await self._fallback_extract_insights(text, insight_types)
                
        except Exception as e:
            logger.error(f"Insight extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_extract_insights(self, text: str, insight_types: List[str], business_context: str, priority: str) -> Dict[str, Any]:
        """AI-powered insight extraction using Azure OpenAI"""
        try:
            insight_list = ", ".join(insight_types)
            
            priority_instructions = {
                "high_level": "Focus on the most important strategic insights only.",
                "detailed": "Provide comprehensive insights with supporting details.",
                "comprehensive": "Extract all possible insights with deep analysis and context."
            }
            
            prompt = f"""
            Extract business insights from the following text.
            Context: {business_context}
            Insight types to focus on: {insight_list}
            Analysis depth: {priority_instructions.get(priority, "detailed analysis")}
            
            Text: {text}
            
            Provide insights in JSON format:
            {{
                "insights": [
                    {{
                        "type": "trend/opportunity/risk/recommendation",
                        "title": "Insight title",
                        "description": "Detailed description",
                        "importance": "high/medium/low",
                        "confidence": 0.9,
                        "supporting_evidence": ["evidence1", "evidence2"],
                        "actionable_items": ["action1", "action2"]
                    }}
                ],
                "key_metrics": {{
                    "metric1": "value1",
                    "metric2": "value2"
                }},
                "summary": "Executive summary of all insights",
                "recommendations": ["recommendation1", "recommendation2"],
                "risk_factors": ["risk1", "risk2"],
                "opportunities": ["opportunity1", "opportunity2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": f"You are an expert business analyst specializing in {business_context}. Extract actionable insights and recommendations."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=1200)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    return {
                        "success": True,
                        "insights": result.get("insights", []),
                        "key_metrics": result.get("key_metrics", {}),
                        "summary": result.get("summary", ""),
                        "recommendations": result.get("recommendations", []),
                        "risk_factors": result.get("risk_factors", []),
                        "opportunities": result.get("opportunities", []),
                        "business_context": business_context,
                        "insight_types": insight_types,
                        "priority": priority,
                        "method": "azure_openai"
                    }
                    
                except json.JSONDecodeError:
                    return await self._fallback_extract_insights(text, insight_types)
            else:
                return await self._fallback_extract_insights(text, insight_types)
                
        except Exception as e:
            logger.error(f"AI insight extraction failed: {e}")
            return await self._fallback_extract_insights(text, insight_types)
    
    async def _compare_texts(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple texts using Azure OpenAI"""
        try:
            texts = arguments.get("texts", [])
            comparison_type = arguments.get("comparison_type", "similarity")
            focus_areas = arguments.get("focus_areas", [])
            output_format = arguments.get("output_format", "narrative")
            
            if len(texts) < 2:
                return {"success": False, "error": "At least 2 texts are required for comparison"}
            
            # Use Azure OpenAI if available
            if llm_client and llm_client.client:
                return await self._ai_compare_texts(texts, comparison_type, focus_areas, output_format)
            else:
                return await self._fallback_compare_texts(texts, comparison_type)
                
        except Exception as e:
            logger.error(f"Text comparison failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_compare_texts(self, texts: List[str], comparison_type: str, focus_areas: List[str], output_format: str) -> Dict[str, Any]:
        """AI-powered text comparison using Azure OpenAI"""
        try:
            focus_instruction = f" Focus particularly on: {', '.join(focus_areas)}." if focus_areas else ""
            
            comparison_instructions = {
                "similarity": "Identify similarities, common themes, and shared concepts.",
                "differences": "Highlight differences, contrasts, and unique aspects of each text.",
                "evolution": "Analyze how ideas or themes evolve across the texts chronologically.",
                "sentiment_shift": "Track changes in sentiment, tone, and emotional content."
            }
            
            # Prepare texts with labels
            labeled_texts = []
            for i, text in enumerate(texts, 1):
                labeled_texts.append(f"Text {i}:\n{text}\n")
            
            combined_texts = "\n".join(labeled_texts)
            
            prompt = f"""
            Compare the following texts using {comparison_type} analysis.
            {comparison_instructions.get(comparison_type, "General comparison")}
            {focus_instruction}
            
            {combined_texts}
            
            Provide comparison results in JSON format:
            {{
                "comparison_type": "{comparison_type}",
                "similarities": ["similarity1", "similarity2"],
                "differences": ["difference1", "difference2"],
                "key_themes": ["theme1", "theme2"],
                "sentiment_analysis": {{
                    "text_1": "positive/negative/neutral",
                    "text_2": "positive/negative/neutral",
                    "overall_trend": "improving/declining/stable"
                }},
                "conclusion": "Overall comparison summary",
                "confidence": 0.9,
                "detailed_analysis": {{
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"],
                    "recommendations": ["rec1", "rec2"]
                }}
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert text analysis specialist. Provide detailed, objective comparisons between texts."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.3, max_tokens=1000)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    return {
                        "success": True,
                        "comparison_type": comparison_type,
                        "similarities": result.get("similarities", []),
                        "differences": result.get("differences", []),
                        "key_themes": result.get("key_themes", []),
                        "sentiment_analysis": result.get("sentiment_analysis", {}),
                        "conclusion": result.get("conclusion", ""),
                        "confidence": result.get("confidence", 0.75),
                        "detailed_analysis": result.get("detailed_analysis", {}),
                        "texts_compared": len(texts),
                        "focus_areas": focus_areas,
                        "output_format": output_format,
                        "method": "azure_openai"
                    }
                    
                except json.JSONDecodeError:
                    return await self._fallback_compare_texts(texts, comparison_type)
            else:
                return await self._fallback_compare_texts(texts, comparison_type)
                
        except Exception as e:
            logger.error(f"AI text comparison failed: {e}")
            return await self._fallback_compare_texts(texts, comparison_type)
    
    async def _analyze_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user query using Azure OpenAI"""
        try:
            query = arguments.get("query", "").strip()
            context = arguments.get("context", "")
            response_type = arguments.get("response_type", "full_analysis")
            include_suggestions = arguments.get("include_suggestions", True)
            
            if not query:
                return {"success": False, "error": "Query is required for analysis"}
            
            # Use Azure OpenAI if available
            if llm_client and llm_client.client:
                return await self._ai_analyze_query(query, context, response_type, include_suggestions)
            else:
                return await self._fallback_analyze_query(query, response_type)
                
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_analyze_query(self, query: str, context: str, response_type: str, include_suggestions: bool) -> Dict[str, Any]:
        """AI-powered query analysis using Azure OpenAI"""
        try:
            context_instruction = f" Context: {context}" if context else ""
            suggestion_instruction = " Include suggestions for improving the query." if include_suggestions else ""
            
            response_instructions = {
                "intent_only": "Focus only on identifying the user's intent.",
                "full_analysis": "Provide comprehensive analysis including intent, complexity, and requirements.",
                "recommendations": "Focus on recommendations and next steps."
            }
            
            prompt = f"""
            Analyze the following user query.{context_instruction}
            {response_instructions.get(response_type, "Provide full analysis")}
            {suggestion_instruction}
            
            Query: "{query}"
            
            Provide analysis in JSON format:
            {{
                "intent": "primary intent (search/analysis/calculation/etc)",
                "complexity": "low/medium/high",
                "query_type": "informational/transactional/navigational",
                "key_entities": ["entity1", "entity2"],
                "intent_confidence": 0.9,
                "clarity_score": 0.8,
                "completeness_score": 0.7,
                "suggested_improvements": ["improvement1", "improvement2"],
                "required_capabilities": ["capability1", "capability2"],
                "expected_output": "description of what user likely expects",
                "processing_approach": "recommended approach to handle this query",
                "estimated_complexity": "processing complexity assessment"
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert query analysis system. Analyze user queries to understand intent and requirements."},
                {"role": "user", "content": prompt}
            ]
            
            response = await llm_client.generate(messages, temperature=0.2, max_tokens=800)
            
            if response:
                try:
                    result = json.loads(response.strip())
                    
                    return {
                        "success": True,
                        "intent": result.get("intent", "unknown"),
                        "complexity": result.get("complexity", "medium"),
                        "query_type": result.get("query_type", "informational"),
                        "key_entities": result.get("key_entities", []),
                        "intent_confidence": result.get("intent_confidence", 0.75),
                        "clarity_score": result.get("clarity_score", 0.7),
                        "completeness_score": result.get("completeness_score", 0.7),
                        "suggested_improvements": result.get("suggested_improvements", []),
                        "required_capabilities": result.get("required_capabilities", []),
                        "expected_output": result.get("expected_output", ""),
                        "processing_approach": result.get("processing_approach", ""),
                        "estimated_complexity": result.get("estimated_complexity", ""),
                        "response_type": response_type,
                        "method": "azure_openai"
                    }
                    
                except json.JSONDecodeError:
                    return await self._fallback_analyze_query(query, response_type)
            else:
                return await self._fallback_analyze_query(query, response_type)
                
        except Exception as e:
            logger.error(f"AI query analysis failed: {e}")
            return await self._fallback_analyze_query(query, response_type)
    
    # Fallback Methods (rule-based/regex)
    
    async def _fallback_summarize(self, text: str, max_length: int, style: str) -> Dict[str, Any]:
        """Fallback summarization using simple text processing"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Simple extractive summarization - take first and important sentences
            important_sentences = []
            
            # Add first sentence
            if sentences:
                important_sentences.append(sentences[0])
            
            # Add sentences with keywords
            keywords = ['important', 'key', 'significant', 'main', 'primary', 'major', 'critical']
            for sentence in sentences[1:]:
                if any(keyword in sentence.lower() for keyword in keywords):
                    important_sentences.append(sentence)
                    if len(' '.join(important_sentences).split()) >= max_length:
                        break
            
            # If still too short, add more sentences
            if len(important_sentences) < 3 and len(sentences) > len(important_sentences):
                for sentence in sentences[len(important_sentences):]:
                    important_sentences.append(sentence)
                    if len(' '.join(important_sentences).split()) >= max_length:
                        break
            
            summary = '. '.join(important_sentences)
            
            return {
                "success": True,
                "summary": summary,
                "key_points": important_sentences[:3],
                "word_count": len(summary.split()),
                "confidence": 60,
                "style": style,
                "method": "extractive_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback summarization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_extract_entities(self, text: str, entity_types: List[str]) -> Dict[str, Any]:
        """Fallback entity extraction using regex patterns"""
        try:
            entities = []
            
            # Simple regex patterns for common entities
            patterns = {
                "person": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                "organization": r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                "location": r'\b[A-Z][a-z]+(?:, [A-Z][a-z]+)*\b',
                "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
                "money": r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            }
            
            for entity_type in entity_types:
                if entity_type in patterns:
                    matches = re.finditer(patterns[entity_type], text)
                    for match in matches:
                        entities.append({
                            "text": match.group(),
                            "type": entity_type,
                            "confidence": 0.7,
                            "start_pos": match.start(),
                            "end_pos": match.end(),
                            "context": text[max(0, match.start()-10):match.end()+10]
                        })
            
            return {
                "success": True,
                "entities": entities,
                "total_found": len(entities),
                "entity_counts": self._count_entity_types(entities),
                "method": "regex_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback entity extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using keyword-based approach"""
        try:
            text_lower = text.lower()
            
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'positive']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'negative', 'poor', 'worst']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                score = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = "negative"
                score = max(-0.8, -0.5 - (negative_count - positive_count) * 0.1)
            else:
                sentiment = "neutral"
                score = 0.0
            
            return {
                "success": True,
                "overall_sentiment": sentiment,
                "confidence": 0.6,
                "sentiment_score": score,
                "emotions": {"joy": max(0, score), "sadness": max(0, -score)},
                "key_phrases": [],
                "reasoning": f"Keyword-based analysis: {positive_count} positive, {negative_count} negative words",
                "method": "keyword_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_extract_insights(self, text: str, insight_types: List[str]) -> Dict[str, Any]:
        """Fallback insight extraction using pattern matching"""
        try:
            insights = []
            
            # Simple pattern-based insights
            if "trend" in insight_types or "trends" in insight_types:
                if any(word in text.lower() for word in ["increase", "growth", "rising", "trending"]):
                    insights.append({
                        "type": "trend",
                        "title": "Growth Pattern Detected",
                        "description": "Text indicates increasing or growth patterns",
                        "importance": "medium",
                        "confidence": 0.6
                    })
            
            if "opportunity" in insight_types or "opportunities" in insight_types:
                if any(word in text.lower() for word in ["opportunity", "potential", "chance", "possibility"]):
                    insights.append({
                        "type": "opportunity",
                        "title": "Potential Opportunity Identified",
                        "description": "Text mentions potential opportunities or possibilities",
                        "importance": "medium",
                        "confidence": 0.6
                    })
            
            return {
                "success": True,
                "insights": insights,
                "key_metrics": {},
                "summary": f"Basic pattern analysis identified {len(insights)} potential insights",
                "recommendations": ["Use AI-powered analysis for deeper insights"],
                "method": "pattern_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback insight extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_compare_texts(self, texts: List[str], comparison_type: str) -> Dict[str, Any]:
        """Fallback text comparison using basic similarity"""
        try:
            # Simple word-based comparison
            word_sets = [set(text.lower().split()) for text in texts]
            
            # Find common words
            common_words = word_sets[0]
            for word_set in word_sets[1:]:
                common_words = common_words.intersection(word_set)
            
            # Find unique words for each text
            unique_words = []
            for i, word_set in enumerate(word_sets):
                unique = word_set - set().union(*[ws for j, ws in enumerate(word_sets) if i != j])
                unique_words.append(list(unique)[:5])  # Limit to 5 unique words
            
            return {
                "success": True,
                "comparison_type": comparison_type,
                "similarities": [f"Common words: {', '.join(list(common_words)[:10])}"],
                "differences": [f"Text {i+1} unique words: {', '.join(words)}" for i, words in enumerate(unique_words)],
                "key_themes": list(common_words)[:5],
                "conclusion": f"Basic word-level comparison of {len(texts)} texts",
                "confidence": 0.5,
                "method": "word_comparison_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback text comparison failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_analyze_query(self, query: str, response_type: str) -> Dict[str, Any]:
        """Fallback query analysis using keyword detection"""
        try:
            query_lower = query.lower()
            
            # Simple intent detection
            intent_patterns = {
                "search": ["search", "find", "look for", "what is", "who is"],
                "analysis": ["analyze", "compare", "evaluate", "assess"],
                "calculation": ["calculate", "compute", "count", "sum"],
                "summary": ["summarize", "summary", "brief", "overview"]
            }
            
            detected_intent = "unknown"
            for intent, patterns in intent_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    detected_intent = intent
                    break
            
            # Simple complexity assessment
            complexity = "low" if len(query.split()) < 5 else "medium" if len(query.split()) < 15 else "high"
            
            return {
                "success": True,
                "intent": detected_intent,
                "complexity": complexity,
                "query_type": "informational",
                "key_entities": [],
                "intent_confidence": 0.6,
                "clarity_score": 0.7,
                "completeness_score": 0.7,
                "suggested_improvements": ["Consider being more specific about your requirements"],
                "required_capabilities": [detected_intent],
                "expected_output": f"Results related to {detected_intent}",
                "processing_approach": f"Use {detected_intent} capabilities",
                "method": "keyword_fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback query analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods
    def _extract_key_points_from_text(self, text: str) -> List[str]:
        """Extract key points from text"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences[:3] if s.strip()]
    
    def _count_entity_types(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count entities by type"""
        counts = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts
    
    def _format_entities(self, entities: List[Dict[str, Any]], output_format: str) -> str:
        """Format entities according to specified format"""
        if output_format == "list":
            return "\n".join([f"- {e['text']} ({e['type']})" for e in entities])
        elif output_format == "highlighted":
            # Simple highlighting format
            return "\n".join([f"**{e['text']}** ({e['type']}) - confidence: {e.get('confidence', 0):.2f}" for e in entities])
        else:  # structured
            return json.dumps(entities, indent=2)
    
    async def _sse_stream(self, session_id: Optional[str]):
        """Generate SSE stream for server-to-client communication"""
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': time.time()})}\n\n"
        
        try:
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

# Create server instance
server = EnhancedNLPSummarizerMCPServer()
app = server.app

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Starting Enhanced NLP Summarizer MCP Server")
    print("=" * 60)
    print(f"🌐 Port: 9103")
    print(f"⚡ Server Name: Enhanced NLP Summarizer Server")
    print(f"🔧 Capabilities: Summarize, Extract Entities, Sentiment Analysis, Insights, Compare, Query Analysis")
    print(f"🤖 Azure OpenAI: Full integration with intelligent fallbacks")
    print(f"📊 MCP Protocol: 2025-06-18")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=9103, log_level="info")