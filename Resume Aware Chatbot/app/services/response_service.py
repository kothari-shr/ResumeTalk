import os
from typing import List, Dict, Any
from email_sender import send_email_to_user

class ResponseService:
    @staticmethod
    def is_unknown_answer(answer: str) -> bool:
        """Detect if the answer indicates unknown information"""
        lower = answer.lower()
        unknown_phrases = ["i don't know", "idk", "not found in", "cannot find"]
        return (
            any(phrase in lower for phrase in unknown_phrases) 
            # or len(answer.strip()) < 1
        )
    
    @staticmethod
    async def handle_unknown_answer(
        question: str,
        session_id: str
    ) -> str:
        """Handle unknown answers by sending a simple email notification."""
        try:
            email_body = f"""Unanswered interview question detected.

Question: {question}

Session: {session_id}
Timestamp: {os.environ.get('TZ', 'UTC')}
"""
            send_email_to_user("Interview Question - Needs Review", email_body)
        except Exception as email_error:
            print(f"Failed to send email notification: {email_error}")

        return "I couldn't find that information in the resume. The question has been forwarded for review."

response_service = ResponseService()