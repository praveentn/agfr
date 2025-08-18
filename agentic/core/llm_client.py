# agentic/core/llm_client.py
import asyncio
import json
from openai import AsyncAzureOpenAI
from typing import Dict, Any, Optional, List
import logging
from .config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            logger.warning("Azure OpenAI credentials not configured")
            self.client = None
            return
            
        try:
            self.client = AsyncAzureOpenAI(
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key
            )
            self.deployment = settings.azure_openai_deployment
            self.max_tokens = settings.llm_max_tokens
            self.temperature = settings.llm_temperature
            logger.info(f"Azure OpenAI client initialized with deployment: {self.deployment}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate response using Azure OpenAI"""
        if not self.client:
            logger.error("Azure OpenAI client not initialized")
            return None
            
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
    
    async def classify_intent(self, query: str) -> str:
        """Classify user intent using Azure OpenAI"""
        if not self.client:
            return "general"
            
        try:
            messages = [
                {
                    "role": "system", 
                    "content": """You are an intent classifier for an agentic workflow system. 
                    Classify the user query into one of these intents:
                    - market_research: research markets, competitors, industry analysis
                    - web_search: find information online
                    - data_analysis: analyze, process, or tabulate data  
                    - calculation: mathematical operations
                    - sql_query: database operations
                    - general: default for unclear intents
                    
                    Respond with ONLY the intent name (lowercase, underscore format)."""
                },
                {"role": "user", "content": f"Classify this query: {query}"}
            ]
            
            response = await self.generate(messages, temperature=0.1, max_tokens=50)
            if response:
                intent = response.strip().lower()
                valid_intents = ["market_research", "web_search", "data_analysis", "calculation", "sql_query", "general"]
                return intent if intent in valid_intents else "general"
            return "general"
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "general"
    
    async def summarize_text(self, text: str, style: str = "brief", max_length: int = 200) -> Dict[str, Any]:
        """Summarize text using Azure OpenAI"""
        if not self.client:
            return {"summary": "Azure OpenAI not available", "key_points": [], "word_count": 0}
            
        try:
            style_prompts = {
                "brief": "Create a brief, concise summary in 2-3 sentences.",
                "detailed": "Create a detailed summary covering main points and insights.",
                "bullet_points": "Create a bullet-point summary with key findings."
            }
            
            prompt = style_prompts.get(style, style_prompts["brief"])
            
            messages = [
                {
                    "role": "system",
                    "content": f"You are a text summarization expert. {prompt} Focus on the most important information and insights."
                },
                {"role": "user", "content": f"Summarize this text:\n\n{text}"}
            ]
            
            response = await self.generate(messages, max_tokens=max_length * 2)
            if response:
                # Extract key points from summary
                key_points = []
                if style == "bullet_points":
                    key_points = [line.strip("• -").strip() for line in response.split('\n') if line.strip().startswith(('•', '-'))]
                else:
                    # Extract sentences as key points
                    sentences = response.split('.')
                    key_points = [s.strip() for s in sentences if len(s.strip()) > 10][:5]
                
                return {
                    "summary": response,
                    "key_points": key_points,
                    "word_count": len(response.split()),
                    "original_length": len(text),
                    "compression_ratio": round(len(response) / len(text), 2) if text else 0
                }
            
            return {"summary": "Failed to generate summary", "key_points": [], "word_count": 0}
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return {"summary": f"Error: {str(e)}", "key_points": [], "word_count": 0}
    
    async def extract_structured_data(self, text: str, fields: List[str]) -> List[Dict[str, Any]]:
        """Extract structured data from text using Azure OpenAI"""
        if not self.client:
            return []
            
        try:
            fields_str = ", ".join(fields)
            messages = [
                {
                    "role": "system",
                    "content": f"""Extract structured data from the text and return it as JSON array.
                    Each item should have these fields: {fields_str}
                    If a field is not available, use "N/A" as the value.
                    Return valid JSON only, no additional text."""
                },
                {"role": "user", "content": text}
            ]
            
            response = await self.generate(messages, temperature=0.1)
            if response:
                try:
                    # Try to parse as JSON
                    data = json.loads(response)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
                except json.JSONDecodeError:
                    # Fallback: extract manually
                    pass
            
            return []
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return []

llm_client = LLMClient()