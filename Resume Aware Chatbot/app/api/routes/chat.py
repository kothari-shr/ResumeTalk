from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import rag_service
from app.services.memory_service import chat_memory
from app.services.response_service import response_service
from app.core.config import settings

router = APIRouter()

@router.post(
    "/chat", 
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process chat message",
    description="Send a question about the resume and get an AI-powered response with source citations"
)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message and return a conversational response string"""
    
    if not rag_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "RAG service not initialized",
                "message": "Please wait and try again",
                "retry_after": 5
            }
        )
    
    try:
        # Get chat history in modern format
        chat_history = chat_memory.get_langchain_format(request.session_id)
        
        # Query RAG chain with latest async pattern
        result = await rag_service.query(request.question, chat_history)
        
    except Exception as e:
        print(f"❌ RAG chain error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Processing failed",
                "message": "Failed to process question. Please try again.",
                "type": str(type(e).__name__)
            }
        )
    
    answer = result.get("answer", "Not Sure").strip()
    
    # Check for unknown answers
    if response_service.is_unknown_answer(answer):
        # notify internally when the model indicates uncertainty
        answer = await response_service.handle_unknown_answer(
            request.question, request.session_id
        )
    
    # Update memory
    if request.session_id:
        chat_memory.add_exchange(request.session_id, request.question, answer)

    return ChatResponse(response=answer)