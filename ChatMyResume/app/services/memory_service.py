from typing import Dict, List, Tuple
from collections import defaultdict
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage

class ChatMemoryService:
    def __init__(self):
        # Modern typing with better structure
        self._memory: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Add a question-answer pair to session memory with size limiting"""
        self._memory[session_id].append((question, answer))
        
        # Use settings-based limit
        max_history = settings.max_history_per_session
        if len(self._memory[session_id]) > max_history:
            # Keep most recent messages
            self._memory[session_id] = self._memory[session_id][-max_history:]
    
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get chat history for a session"""
        return self._memory.get(session_id, [])
    
    def clear_session(self, session_id: str) -> bool:
        """Clear chat history for a session. Returns True if session existed."""
        if session_id in self._memory:
            del self._memory[session_id]
            return True
        return False
    
    def get_langchain_format(self, session_id: str) -> List:
        """Get history in LangChain message format for MessagesPlaceholder"""
        history = self.get_history(session_id)
        messages = []
        for question, answer in history:
            messages.append(HumanMessage(content=question))
            messages.append(AIMessage(content=answer))
        return messages
    
    def get_all_sessions(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get all active sessions (new method)"""
        return dict(self._memory)
    
    def session_count(self) -> int:
        """Get number of active sessions"""
        return len(self._memory)

# Global instance
chat_memory = ChatMemoryService()