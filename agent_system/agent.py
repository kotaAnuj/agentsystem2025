import os
import json
import importlib
import re
from typing import Dict, Any, List, Optional
from memory import Memory
from llm_provider import get_llm_provider

class Agent:
    """
    Core Agent class that processes user queries using LLM and configured tools
    """
    
    def __init__(self, config_path: str, llm_provider: str = "openai"):
        """Initialize the agent with a configuration file path and LLM provider"""
        self.config = self._load_config(config_path)
        self.name = self.config.get("agent_name", "Assistant")
        self.tools = self._load_tools(self.config.get("config", {}).get("tools", []))
        self.memory = Memory(max_items=10, enabled=self.config.get("config", {}).get("memory", False))
        self.llm_provider_name = llm_provider
        self.llm = get_llm_provider(llm_provider)
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _load_tools(self, tool_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load tool definitions from tool configuration files"""
        tools = {}
        
        for tool_name in tool_names:
            tool_config_path = f"tools/{tool_name}.json"
            if not os.path.exists(tool_config_path):
                print(f"Warning: Tool config not found: {tool_config_path}")
                continue
                
            with open(tool_config_path, 'r') as f:
                tool_config = json.load(f)
                tools[tool_name] = tool_config
                
        return tools
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool function with the given parameters"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
            
        tool_config = self.tools[tool_name]
        function_path = tool_config.get("function")
        
        if not function_path:
            return f"Error: Function path not defined for tool '{tool_name}'"
            
        try:
            module_name, function_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            return function(**parameters)
        except (ImportError, AttributeError) as e:
            return f"Error loading tool function: {str(e)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def _create_system_prompt(self) -> str:
        """Create system prompt from config"""
        config_data = self.config.get("config", {})
        backstory = config_data.get("backstory", "")
        task = config_data.get("task", "")
        prompt_template = config_data.get("prompt_template", "")
        think = config_data.get("think", "")
        
        if prompt_template:
            # Use custom template if provided
            system_prompt = prompt_template.format(
                agent_name=self.name,
                backstory=backstory,
                task=task
            )
        else:
            # Default template
            system_prompt = f"""You are {self.name}. {backstory}
Your task is to {task}.

You have access to the following tools:
"""
            # Add tools information
            for tool_name, tool_config in self.tools.items():
                system_prompt += f"- {tool_name}: {tool_config.get('description', '')}\n"
            
            if think:
                system_prompt += f"\nThinking process: {think}\n"
        
        system_prompt += """
To use a tool, use the following format:
```
{
  "tool": "tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

First think about the request, then decide if you need to use a tool.
If you need to use a tool, output ONLY the JSON above.
After receiving tool results, respond to the user naturally.
"""
        return system_prompt
    
    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Format tools in format appropriate for the LLM provider"""
        return self.llm.format_tools(self.tools)
    
    def process_query(self, query: str) -> str:
        """Process a user query using LLM and tools"""
        try:
            # Prepare messages for the LLM
            messages = [{"role": "system", "content": self._create_system_prompt()}]
            
            # Add memory if enabled
            if self.memory.is_enabled():
                messages.extend(self.memory.get_messages())
                
            # Add the current query
            messages.append({"role": "user", "content": query})
            self.memory.add("user", query)
            
            # Get tool recommendations from LLM
            tool_response = self.llm.get_response(
                messages=messages,
                tools=self._format_tools_for_llm()
            )
            
            # Check if LLM wants to use a tool
            tool_call = self.llm.extract_tool_call(tool_response)
            
            if tool_call:
                # Extract tool details
                tool_name = tool_call.get("tool")
                parameters = tool_call.get("parameters", {})
                
                # Execute the tool
                tool_result = self._execute_tool(tool_name, parameters)
                
                # Add tool response to context
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(tool_call)
                })
                
                messages.append({
                    "role": "system",
                    "content": f"Tool result: {tool_result}"
                })
                
                # Get final response after tool execution
                final_response = self.llm.get_response(
                    messages=messages,
                    tools=None  # No tools on final response
                )
                
                self.memory.add("assistant", final_response)
                return final_response
            else:
                # If no tool call, just return the response
                self.memory.add("assistant", tool_response)
                return tool_response
                
        except Exception as e:
            return f"Error processing query: {str(e)}"


def create_agent(config_path: str, llm_provider: str = "openai"):
    """Create and return an Agent instance with specified LLM provider"""
    return Agent(config_path, llm_provider)

def run_agent(agent: Agent, query: str) -> str:
    """Run the agent with a query and return the response"""
    return agent.process_query(query)