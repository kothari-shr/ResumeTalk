import os
from typing import Optional, Tuple, Dict, List, Any
from app.core.config import settings
from resume_loader import load_and_split_resume
from rag_chain import bootstrap_rag

class RAGService:
    def __init__(self):
        self.chain = None
        self.vectorstore = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize RAG chain and vectorstore using latest LangChain patterns"""
        if self._initialized:
            return
        
        if not settings.resume_path or not os.path.isfile(settings.resume_path):
            raise RuntimeError(f"RESUME_PATH not set or file not found: {settings.resume_path}")
        
        try:
            docs = load_and_split_resume(settings.resume_path)
            if not docs:
                raise RuntimeError("No text could be extracted from the resume PDF.")
            
            # Use the latest bootstrap_rag function
            self.chain, self.vectorstore = bootstrap_rag(docs)
            self._initialized = True
            print(f"✅ RAG chain initialized with {len(docs)} documents")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG chain: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if RAG service is ready"""
        return self._initialized and self.chain is not None
    
    async def query(self, question: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Query using latest LangChain invoke pattern"""
        if not self.is_ready():
            raise RuntimeError("RAG service not initialized")
        
        try:
            # Use invoke() method with latest LangChain
            # the LCEL chain expects the user text under the key "input"
            # (rag_chain.py references x["input"]). Provide both keys for
            # backward compatibility but ensure "input" is present.
            result = await self.chain.ainvoke({
                "input": question,
                "question": question,
                "chat_history": chat_history
            })
            return result
        except Exception as e:
            print(f"RAG chain error: {e}")
            # Fallback to sync invoke if async not available
            try:
                result = self.chain.invoke({
                    "input": question,
                    "question": question,
                    "chat_history": chat_history
                })
                return result
            except Exception as sync_e:
                print(f"Sync invoke also failed: {sync_e}")
                raise e
    
    def format_sources(self, source_docs: List[Any]) -> List[Dict[str, str]]:
        """Format source documents for response using latest structure"""
        sources = []
        for doc in source_docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                src = doc.metadata.get("source", "resume.pdf")
                page = doc.metadata.get("page")
                sources.append({
                    "source": os.path.basename(src), 
                    "page": str(page) if page is not None else "unknown"
                })
        return sources

# Global instance
rag_service = RAGService()