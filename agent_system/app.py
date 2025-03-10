# Import the required modules
from agent import create_agent, run_agent
from tool import ToolManager
from memory import Memory
from llm_provider import get_llm_provider
import os

# Ensure we have a config file
config_path = "agent_config.json"
if not os.path.exists(config_path):
    # Create default configuration if it doesn't exist
    ToolManager.setup_basic_config()

# Set API key for OpenAI (assuming OpenAI as default provider)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Replace with actual key

# Create the agent
agent = create_agent(config_path, "openai")

# Process a query and print the response
query = "What is 2 + 2?"
response = run_agent(agent, query)
print(f"{agent.name}: {response}")
