import os
import json
import re
import importlib.util
from typing import Dict, Any, List, Optional

class LLMProvider:
    """Base class for LLM providers"""
    
    def format_tools(self, tools: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for the LLM provider
        
        Args:
            tools: Dictionary of tool configurations
        
        Returns:
            Formatted tools list for the LLM
        """
        raise NotImplementedError("Subclasses must implement format_tools")
    
    def get_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
        """Get response from the LLM
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of formatted tools
        
        Returns:
            Response from the LLM
        """
        raise NotImplementedError("Subclasses must implement get_response")
    
    def extract_tool_call(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response
        
        Args:
            response_text: Response text from the LLM
        
        Returns:
            Dictionary with tool name and parameters if a tool call is found
        """
        try:
            # Look for JSON format in the response
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text, re.DOTALL)
            if not json_match:
                # Try without code blocks in case LLM just returned JSON directly
                json_match = re.search(r'^({[\s\S]*})$', response_text.strip(), re.DOTALL)
                
            if json_match:
                json_str = json_match.group(1)
                tool_call = json.loads(json_str)
                if "tool" in tool_call and "parameters" in tool_call:
                    return tool_call
        except Exception as e:
            print(f"Error extracting tool call: {str(e)}")
        
        return None


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation"""
    
    def __init__(self, model="gpt-4"):
        """Initialize the OpenAI provider"""
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set in environment variables")
    
    def format_tools(self, tools: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI function calling format"""
        formatted_tools = []
        
        for tool_name, tool_config in tools.items():
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_config.get("description", ""),
                    "parameters": tool_config.get("parameters", {})
                }
            }
            formatted_tools.append(formatted_tool)
            
        return formatted_tools
    
    def get_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
        """Get response from OpenAI"""
        if not self.api_key:
            return "Error: OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
        
        try:
            # Check if openai module is available
            if importlib.util.find_spec("openai") is None:
                return "Error: OpenAI module not installed. Please install it with 'pip install openai'"
            
            import openai
            
            client = openai.Client(api_key=self.api_key)
            
            args = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7
            }
            
            # Add tools if provided
            if tools:
                args["tools"] = tools
                args["tool_choice"] = "auto"
            
            response = client.chat.completions.create(**args)
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, model="claude-3-opus-20240229"):
        """Initialize the Anthropic provider"""
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            print("Warning: ANTHROPIC_API_KEY not set in environment variables")
    
    def format_tools(self, tools: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Claude (returned as empty list since Claude uses prompt)"""
        # Claude doesn't support tool calling like OpenAI, so we'll handle it with special prompting
        # The actual tools info is added to the system prompt in Agent class
        return []
    
    def get_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
        """Get response from Anthropic"""
        if not self.api_key:
            return "Error: Anthropic API key not set. Please set the ANTHROPIC_API_KEY environment variable."
        
        try:
            # Check if anthropic module is available
            if importlib.util.find_spec("anthropic") is None:
                return "Error: Anthropic module not installed. Please install it with 'pip install anthropic'"
            
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Extract system prompt from messages
            system_prompt = ""
            claude_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    # System messages are handled differently in Claude
                    if not claude_messages:
                        # If this is the first message, it will be the system prompt
                        system_prompt = msg["content"]
                    else:
                        # Otherwise, just add it as an assistant message
                        claude_messages.append({
                            "role": "assistant",
                            "content": f"System message: {msg['content']}"
                        })
                elif msg["role"] == "user":
                    claude_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    claude_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            
            # Create Claude message request
            response = client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=claude_messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"


def get_llm_provider(provider_name: str) -> LLMProvider:
    """Get the appropriate LLM provider based on name
    
    Args:
        provider_name: Name of the provider ("openai" or "anthropic")
    
    Returns:
        LLMProvider instance
    
    Raises:
        ValueError: If an unsupported provider is specified
    """
    if provider_name.lower() == "openai":
        return OpenAIProvider()
    elif provider_name.lower() == "anthropic":
        return AnthropicProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")