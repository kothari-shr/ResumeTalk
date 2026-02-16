import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging
from app.core.config import settings
from resume_loader import load_and_split_resume
from rag_chain import bootstrap_rag
from app.core.intent_classifier import classify_intent

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.chain = None
        self._initialized = False
        self._meta_filename = "meta.json"
    
    def _vectorstore_dir(self) -> Path:
        return Path(settings.vectorstore_path).resolve()
    
    def _meta_path(self) -> Path:
        return self._vectorstore_dir() / self._meta_filename
    
    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def _read_meta(self) -> Dict[str, Any]:
        try:
            p = self._meta_path()
            if p.is_file():
                return json.loads(p.read_text())
        except Exception:
            pass
        return {}
    
    def _write_meta(self, data: Dict[str, Any]) -> None:
        try:
            d = self._vectorstore_dir()
            d.mkdir(parents=True, exist_ok=True)
            (d / self._meta_filename).write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to write meta: {e}")
    
    def delete_vectorstore(self) -> bool:
        """Delete the FAISS vectorstore directory if it exists."""
        vs_dir = self._vectorstore_dir()
        if vs_dir.exists():
            try:
                shutil.rmtree(vs_dir)
                print(f"🗑️ Deleted vectorstore at {vs_dir}")
                return True
            except Exception as e:
                print(f"❌ Failed to delete vectorstore at {vs_dir}: {e}")
                return False
        return False
    
    async def initialize(self) -> None:
        """Initialize RAG chain and vectorstore using latest LangChain patterns"""
        if self._initialized:
            return
        
        if not settings.resume_path or not os.path.isfile(settings.resume_path):
            raise RuntimeError(f"RESUME_PATH not set or file not found: {settings.resume_path}")
        
        try:
            # If index exists but resume changed, wipe it to force rebuild
            current_hash = self._sha256_file(settings.resume_path)
            meta = self._read_meta()
            if meta.get("resume_sha256") and meta.get("resume_sha256") != current_hash:
                print("♻️ Resume changed detected. Rebuilding vectorstore...")
                logger.info("Resume change detected. Rebuilding vectorstore.")
                self.delete_vectorstore()
            
            docs = load_and_split_resume(settings.resume_path)
            if not docs:
                raise RuntimeError("No text could be extracted from the resume PDF.")
            
            # Use the latest bootstrap_rag function
            self.chain = bootstrap_rag(docs)
            self._initialized = True
            print(f"✅ RAG chain initialized with {len(docs)} documents")
            logger.info("RAG chain initialized with %s documents", len(docs))
            # Persist meta for change detection next time
            self._write_meta({
                "resume_sha256": current_hash,
                "docs_count": len(docs)
            })
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG chain: {e}")
            logger.exception("Failed to initialize RAG chain")
            raise
    
    def is_ready(self) -> bool:
        """Check if RAG service is ready"""
        return self._initialized and self.chain is not None
    
    async def query(self, question: str, chat_history: List) -> Dict[str, Any]:
        """Query using latest LangChain invoke pattern
        
        Args:
            question: User's question text
            chat_history: List of LangChain message objects (HumanMessage, AIMessage)
        """
        if not self.is_ready():
            raise RuntimeError("RAG service not initialized")
        
        # Classify intent to obtain optional retrieval filters (section-based)
        intent = classify_intent(question)
        filters = intent.get("normalized_filters", {}) if intent else {}
        print(f"🧠 Classified intent: {intent.get('target_section', 'general')}, filters: {filters}")
        logger.info(
            "Classified intent: %s, filters: %s",
            intent.get("target_sections", []) if intent else [],
            filters,
        )
        try:
            # Use invoke() method with latest LangChain.
            # Our chain currently reads the user text from "question", but
            # many LangChain components and community prompts expect "input".
            # Provide both keys for maximum compatibility.
            result = await self.chain.ainvoke({
                "input": question,
                "question": question,
                "chat_history": chat_history,
                "filters": filters,
            })
            return result
        except Exception as e:
            logger.warning("RAG chain async error; trying sync invoke", exc_info=e)
            # Fallback to sync invoke if async not available
            try:
                result = self.chain.invoke({
                    "input": question,
                    "question": question,
                    "chat_history": chat_history,
                    "filters": filters,
                })
                return result
            except Exception as sync_e:
                logger.exception("Sync invoke also failed")
                raise e

    async def rebuild(self, force_delete: bool = True) -> None:
        """Rebuild the RAG chain, optionally deleting the vectorstore first."""
        try:
            if force_delete:
                self.delete_vectorstore()
            # Reset state and re-init
            self.chain = None
            self._initialized = False
            await self.initialize()
        except Exception:
            # Keep state consistent on failure
            self.chain = None
            self._initialized = False
            raise

# Global instance
rag_service = RAGService()