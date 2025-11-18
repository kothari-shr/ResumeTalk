from fastapi import APIRouter, HTTPException, status
from app.models.schemas import SessionHistoryResponse, SessionClearResponse
from app.services.memory_service import chat_memory

router = APIRouter()

@router.get(
    "/sessions/{session_id}/history", 
    response_model=SessionHistoryResponse,
    summary="Get session history",
    description="Retrieve chat history for a specific session"
)
async def get_chat_history(session_id: str):
    """Get chat history for a specific session"""
    history_tuples = chat_memory.get_history(session_id)
    
    # Convert tuples to dict format for better API response
    history_dicts = [
        {"question": q, "answer": a} 
        for q, a in history_tuples
    ]
    
    return SessionHistoryResponse(
        session_id=session_id,
        history=history_dicts,
        count=len(history_dicts)
    )

@router.delete(
    "/sessions/{session_id}", 
    response_model=SessionClearResponse,
    summary="Clear session",
    description="Clear chat history for a specific session"
)
async def clear_session(session_id: str):
    """Clear chat history for a specific session"""
    existed = chat_memory.clear_session(session_id)
    
    return SessionClearResponse(
        message=f"Session {session_id} {'cleared' if existed else 'not found'}",
        success=existed
    )

@router.get(
    "/sessions",
    summary="List active sessions",
    description="Get list of all active sessions"
)
async def list_sessions():
    """Get list of all active sessions"""
    sessions = chat_memory.get_all_sessions()
    return {
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(history)
            }
            for sid, history in sessions.items()
        ],
        "total_sessions": len(sessions)
    }

@router.post(
    "/sessions/cleanup",
    summary="Cleanup inactive sessions",
    description="Manually trigger cleanup of inactive sessions"
)
async def cleanup_sessions():
    """Manually trigger cleanup of inactive sessions"""
    removed_count = chat_memory._cleanup_inactive_sessions()
    active_count = chat_memory.session_count()
    return {
        "removed_sessions": removed_count,
        "active_sessions": active_count,
        "message": f"Cleaned up {removed_count} inactive session(s). {active_count} active session(s) remaining."
    }