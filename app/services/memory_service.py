from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import time
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage

class ChatMemoryService:
    def __init__(self, session_timeout_minutes: int = 30, cleanup_interval_seconds: int = 60):
        # Modern typing with better structure
        self._memory: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Track last activity time for each session
        self._last_activity: Dict[str, datetime] = {}
        self._session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_interval_seconds = max(5, int(cleanup_interval_seconds))

        # Background cleanup thread (daemon)
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._run_cleanup_loop, daemon=True)
        # Start only once per process
        if not self._cleanup_thread.is_alive():
            try:
                self._cleanup_thread.start()
            except RuntimeError:
                # In rare cases, thread may already be started in hot-reload contexts
                pass
    
    def _cleanup_inactive_sessions(self) -> int:
        """Remove sessions that have been inactive for too long. Returns count of removed sessions."""
        now = datetime.now()
        inactive_sessions = [
            session_id for session_id, last_time in self._last_activity.items()
            if now - last_time > self._session_timeout
        ]
        
        for session_id in inactive_sessions:
            if session_id in self._memory:
                del self._memory[session_id]
            del self._last_activity[session_id]
        
        return len(inactive_sessions)

    def _run_cleanup_loop(self) -> None:
        """Periodically cleanup inactive sessions in the background."""
        while not self._stop_event.is_set():
            try:
                self._cleanup_inactive_sessions()
            except Exception:
                # Never let the cleanup loop crash the process
                pass
            # Sleep in small chunks to be responsive to stop
            total = 0
            while total < self._cleanup_interval_seconds and not self._stop_event.is_set():
                time.sleep(1)
                total += 1

    def stop(self) -> None:
        """Signal the background cleanup thread to stop (optional)."""
        self._stop_event.set()
    
    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Add a question-answer pair to session memory with size limiting"""
        # Update activity timestamp
        self._last_activity[session_id] = datetime.now()
        
        # Cleanup inactive sessions periodically
        self._cleanup_inactive_sessions()
        
        self._memory[session_id].append((question, answer))
        
        # Use settings-based limit
        max_history = settings.max_history_per_session
        if len(self._memory[session_id]) > max_history:
            # Keep most recent messages
            self._memory[session_id] = self._memory[session_id][-max_history:]
    
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get chat history for a session"""
        # Update activity timestamp when accessing history
        if session_id in self._memory:
            self._last_activity[session_id] = datetime.now()
        return self._memory.get(session_id, [])
    
    def clear_session(self, session_id: str) -> bool:
        """Clear chat history for a session. Returns True if session existed."""
        existed = (session_id in self._memory) or (session_id in self._last_activity)
        if session_id in self._memory:
            del self._memory[session_id]
        if session_id in self._last_activity:
            del self._last_activity[session_id]
        return existed
    
    def get_langchain_format(self, session_id: str) -> List:
        """Get history in LangChain message format for MessagesPlaceholder"""
        # Update activity timestamp
        if session_id in self._memory:
            self._last_activity[session_id] = datetime.now()
        
        history = self.get_history(session_id)
        messages = []
        for question, answer in history:
            messages.append(HumanMessage(content=question))
            messages.append(AIMessage(content=answer))
        return messages
    
    def get_all_sessions(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get all active sessions (new method)"""
        # Cleanup before returning
        self._cleanup_inactive_sessions()
        return dict(self._memory)
    
    def session_count(self) -> int:
        """Get number of active sessions"""
        # Cleanup before counting
        self._cleanup_inactive_sessions()
        return len(self._memory)
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get detailed info about a session"""
        if session_id not in self._memory:
            return {"exists": False}
        
        return {
            "exists": True,
            "message_count": len(self._memory[session_id]),
            "last_activity": self._last_activity.get(session_id),
            "is_active": session_id in self._last_activity and 
                        (datetime.now() - self._last_activity[session_id]) < self._session_timeout
        }

# Global instance - timeout and cleanup interval pulled from settings when available
try:
    chat_memory = ChatMemoryService(
        session_timeout_minutes=settings.session_timeout_minutes,
        cleanup_interval_seconds=getattr(settings, "cleanup_interval_seconds", 60),
    )
except:
    # Fallback if settings not loaded yet
    chat_memory = ChatMemoryService(session_timeout_minutes=30, cleanup_interval_seconds=60)