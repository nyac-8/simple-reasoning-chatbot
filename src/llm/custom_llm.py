"""
Custom LLM wrapper using langchain-google-genai.
Provides a lean interface for text generation, structured output, and tool calling.
"""

import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class CustomLLM:
    """
    Lean LLM wrapper using langchain-google-genai.
    
    Core methods:
    1. generate_content: Text generation (str -> str)
    2. get_structured_output: JSON output with schema (str, dict -> str)
    3. tool_calls: Function calling with tools (str, tools -> str)
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        **kwargs
    ):
        """Initialize with model configuration."""
        self.model_name = model
        self.temperature = temperature
        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"CustomLLM initialized: model={model}, temp={temperature}")
    
    def generate_content(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt string
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def get_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate structured JSON output.
        
        Args:
            prompt: Input prompt string
            schema: JSON schema dict (Vertex AI format)
            **kwargs: Additional parameters
        
        Returns:
            JSON string
        """
        try:
            # Configure model for JSON output
            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema
            )
            
            messages = [HumanMessage(content=prompt)]
            response = model.invoke(messages, **kwargs)
            
            # Response is already JSON formatted
            return response.content
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise
    
    def tool_calls(
        self,
        prompt: str,
        tools: List[BaseTool],
        **kwargs
    ) -> str:
        """
        Generate with tool calling.
        
        Args:
            prompt: Input prompt string
            tools: List of LangChain BaseTool objects
            **kwargs: Additional parameters
        
        Returns:
            JSON string with response and tool calls
        """
        try:
            # Bind tools to model
            model_with_tools = self.model.bind_tools(tools)
            
            messages = [HumanMessage(content=prompt)]
            response = model_with_tools.invoke(messages, **kwargs)
            
            result = {
                "text": response.content,
                "tool_calls": []
            }
            
            # Extract tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    result["tool_calls"].append({
                        "name": tool_call["name"],
                        "args": tool_call["args"]
                    })
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Tool calling failed: {e}")
            raise