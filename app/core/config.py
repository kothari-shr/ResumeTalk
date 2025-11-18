import os
from pathlib import Path
from typing import List, ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Load .env located next to this config file to avoid depending on the process cwd
    # Look for the repository-level .env in the project root (two levels up from this file)
    env_path: ClassVar[Path] = Path(__file__).resolve().parents[2] / ".env"
    model_config = SettingsConfigDict(
        env_file=str(env_path),
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Settings
    app_name: str = Field(default="Resume Chatbot API", description="Application name")
    version: str = Field(default="1.1.0", description="API version")
    host: str = Field(default="0.0.0.0", description="Host address")
    port: int = Field(default=8000, description="Port number")
    debug: bool = Field(default=True, description="Debug mode")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        description="Allowed CORS origins"
    )
    
    # Resume Settings
    resume_path: str = Field(..., description="Path to resume PDF file")
    
    # LLM Settings
    llm_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature")
    embeddings_model: str = Field(default="text-embedding-3-small", description="Embeddings model")
    use_local_embeddings: bool = Field(default=False, description="Use local HuggingFace embeddings")
    
    # Retrieval Settings
    retriever_k: int = Field(default=4, ge=1, le=20, description="Number of documents to retrieve")
    retriever_fetch_k: int = Field(default=12, ge=1, le=50, description="Number of docs for MMR")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR lambda parameter")
    
    # Memory Settings
    max_history_per_session: int = Field(default=50, ge=1, description="Max chat history per session")
    session_timeout_minutes: int = Field(default=30, ge=5, le=1440, description="Session inactivity timeout in minutes")
    cleanup_interval_seconds: int = Field(default=60, ge=5, le=3600, description="Background cleanup interval in seconds")
    
    # Email Settings (optional)
    email_address: str = Field(default="", description="Email for notifications")
    email_app_password: str = Field(default="", description="Email app password")
    smtp_server: str = Field(default="smtp.gmail.com", description="SMTP server")
    smtp_port: int = Field(default=465, description="SMTP port")

settings = Settings()