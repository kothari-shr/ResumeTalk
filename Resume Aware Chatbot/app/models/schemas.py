from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Interview question",
        examples=["What programming languages does this candidate know?"]
    )
    session_id: str = Field(
        min_length=1,
        max_length=100, 
        description="Session identifier",
        examples=["user123", "interviewer1"]
    )

class ChatResponse(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    response: str = Field(description="AI response to the question")

class HealthResponse(BaseModel):
    status: str = Field(description="Service health status")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")

class SessionHistoryResponse(BaseModel):
    session_id: str = Field(description="Session identifier")
    history: List[Dict[str, str]] = Field(description="Chat history as list of exchanges")
    count: int = Field(description="Number of exchanges in history")

class SessionClearResponse(BaseModel):
    message: str = Field(description="Operation result message")
    success: bool = Field(description="Whether operation succeeded")