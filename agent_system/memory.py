from typing import Dict, List, Any

class Memory:
    """Manages conversation memory for the agent"""
    
    def __init__(self, max_items: int = 10, enabled: bool = True):
        """Initialize the memory manager
        
        Args:
            max_items: Maximum number of messages to keep in memory
            enabled: Whether memory is enabled
        """
        self.messages = []
        self.max_items = max_items
        self.enabled = enabled
    
    def add(self, role: str, content: str) -> None:
        """Add a message to memory
        
        Args:
            role: The role of the message sender (user/assistant/system)
            content: The message content
        """
        if not self.enabled:
            return
            
        self.messages.append({"role": role, "content": content})
        
        # Keep memory to max size
        if len(self.messages) > self.max_items:
            self.messages = self.messages[-self.max_items:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in memory
        
        Returns:
            List of message dictionaries with role and content
        """
        return self.messages
    
    def clear(self) -> None:
        """Clear all messages from memory"""
        self.messages = []
    
    def is_enabled(self) -> bool:
        """Check if memory is enabled
        
        Returns:
            Boolean indicating if memory is enabled
        """
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable memory
        
        Args:
            enabled: Boolean to enable/disable memory
        """
        self.enabled = enabled