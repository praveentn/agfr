# ============================================================================
# agentic/core/llm_client.py
import asyncio
from openai import AsyncAzureOpenAI
from typing import Dict, Any, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        config = settings.get_azure_openai_config()
        self.client = AsyncAzureOpenAI(
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
            api_key=config["api_key"]
        )
        self.deployment = config["deployment"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
    
    async def generate(self, messages: list, **kwargs) -> Optional[str]:
        """Generate response using Azure OpenAI"""
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

llm_client = LLMClient()

