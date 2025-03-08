import os
import json
from typing import Dict, Any, List

class ToolManager:
    """Utility class to create and manage tool configurations"""
    
    @staticmethod
    def create_tool_config(
        tool_name: str,
        description: str,
        function_path: str,
        parameters: Dict[str, Any],
        required_params: List[str] = None,
        keywords: List[str] = None
    ) -> None:
        """Create a tool configuration file"""
        os.makedirs("tools", exist_ok=True)
        
        tool_config = {
            "name": tool_name,
            "description": description,
            "function": function_path,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required_params or []
            },
            "keywords": keywords or []
        }
        
        with open(f"tools/{tool_name}.json", 'w') as f:
            json.dump(tool_config, f, indent=2)
        
        print(f"Tool configuration created: tools/{tool_name}.json")
    
    @staticmethod
    def register_basic_tools():
        """Register some basic tools to get started"""
        # Create tools directory if it doesn't exist
        os.makedirs("tools", exist_ok=True)
        
        # Create a simple calculator tool
        ToolManager.create_tool_config(
            tool_name="calculator",
            description="Evaluate a mathematical expression",
            function_path="tools.calculator.evaluate",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g. '2 + 2')"
                }
            },
            required_params=["expression"]
        )
        
        # Create calculator tool implementation
        os.makedirs("tools/calculator", exist_ok=True)
        
        # Create __init__.py to make it a proper package
        with open("tools/calculator/__init__.py", 'w') as f:
            f.write("# Calculator tool package")
        
        # Create the calculator implementation
        with open("tools/calculator/evaluate.py", 'w') as f:
            f.write("""import math

def evaluate(expression):
    \"\"\"
    Safely evaluate a mathematical expression
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float or str: The result of the evaluation, or an error message
    \"\"\"
    try:
        # Create a safe environment with only math functions
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow
        }
        
        # Add all math module functions
        for name in dir(math):
            if not name.startswith('__'):
                safe_dict[name] = getattr(math, name)
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return result
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
""")
        
        print("Calculator tool registered successfully!")
    
    @staticmethod
    def setup_basic_config():
        """Set up a simple agent configuration with basic tools"""
        # Register basic tools
        ToolManager.register_basic_tools()
        
        # Create agent config
        agent_config = {
            "agent_name": "MathAssistant",
            "config": {
                "backstory": "I am a helpful assistant with math skills.",
                "task": "help users solve mathematical problems using my calculator tool when needed",
                "tools": ["calculator"],
                "memory": True
            }
        }
        
        with open("agent_config.json", 'w') as f:
            json.dump(agent_config, f, indent=2)
        
        print("Basic agent configuration set up successfully!")