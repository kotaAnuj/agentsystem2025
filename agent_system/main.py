import os
from agent import create_agent, run_agent
from tool import ToolManager

def main():
    """Run the agent in interactive mode"""
    # Check for config file
    config_path = "agent_config.json"
    
    # If config doesn't exist, create it
    if not os.path.exists(config_path):
        print("Config file not found. Setting up basic configuration...")
        ToolManager.setup_basic_config()
    
    # Determine LLM provider
    provider = input("Choose LLM provider (openai/anthropic) [default: openai]: ").lower()
    if not provider:
        provider = "openai"
    
    if provider == "openai":
        # Check for OpenAI API key
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "anthropic":
        # Check for Anthropic API key
        if "ANTHROPIC_API_KEY" not in os.environ:
            api_key = input("Enter your Anthropic API key: ")
            os.environ["ANTHROPIC_API_KEY"] = api_key
    else:
        print(f"Unsupported LLM provider: {provider}")
        return
    
    # Create an agent
    agent = create_agent(config_path, provider)
    
    # Interactive mode
    print(f"Agent {agent.name} initialized. Type 'exit' to quit.")
    print("=" * 50)
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
            
        response = run_agent(agent, query)
        print(f"\n{agent.name}: {response}")

if __name__ == "__main__":
    main()