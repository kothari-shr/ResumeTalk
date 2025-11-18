"""
Streamlit interface for ChatMyResume Chatbot
Deploy this on Streamlit Cloud or Hugging Face Spaces
"""

import streamlit as st
import os
from pathlib import Path
from typing import List
import asyncio
import uuid

# Set page config
st.set_page_config(
    page_title="Resume Chatbot",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Import services
from app.services.rag_service import rag_service
from app.services.memory_service import chat_memory
from app.core.config import settings

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-doc {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # Generate unique session ID for this user/browser tab
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Register session activity (heartbeat) - this keeps the session alive
# This runs on every page interaction/rerun
if "session_id" in st.session_state:
    # Touch the session to update its last activity time
    _ = chat_memory.get_history(st.session_state.session_id)

async def initialize_rag():
    """Initialize RAG service"""
    if not st.session_state.initialized:
        with st.spinner("🔄 Loading resume and initializing AI..."):
            try:
                await rag_service.initialize()
                st.session_state.initialized = True
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                return False
    return True

async def get_response(question: str):
    """Get response from RAG service"""
    try:
        # Get chat history
        chat_history = chat_memory.get_langchain_format(st.session_state.session_id)
        
        # Query RAG
        result = await rag_service.query(question, chat_history)
        
        answer = result.get("answer", "I'm not sure about that.").strip()
        sources = result.get("source_documents", [])
        
        # Store in memory
        chat_memory.add_exchange(st.session_state.session_id, question, answer)
        
        return answer, sources
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, []

def display_sources(sources):
    """Display source documents"""
    if sources:
        with st.expander("📄 View Sources", expanded=False):
            for i, doc in enumerate(sources, 1):
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                st.markdown(f"""
                <div class="source-doc">
                    <strong>Source {i}:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)

def main():
    # Header
    st.title("💼  ChatMyResume Chatbot")
    st.markdown("Ask me anything about the resume!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info(f"**Model:** {settings.llm_model}")
        
        # Show session ID with copy button
        st.markdown("**Session ID:**")
        st.code(st.session_state.session_id, language=None)
        st.caption("🔒 Your unique session - each browser tab gets its own ID")
        
        if st.button("🗑️ Clear Chat History"):
            # Clear messages in UI
            st.session_state.messages = []
            # Clear session from memory service
            chat_memory.clear_session(st.session_state.session_id)
            # Generate new session ID
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"
            st.success("Chat history cleared! New session started.")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot uses:
        - 🤖 RAG (Retrieval Augmented Generation)
        - 🧠 LangChain
        - 📚 FAISS Vector Store
        - 💬 Conversational Memory
        """)
        
        # Show resume status
        if os.path.exists(settings.resume_path):
            st.success(f"✅ Resume loaded")
        else:
            st.error(f"❌ Resume not found")
    
    # Initialize RAG
    if asyncio.run(initialize_rag()):
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    display_sources(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the resume..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = asyncio.run(get_response(prompt))
                    
                    if answer:
                        st.markdown(answer)
                        display_sources(sources)
                        
                        # Store assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
    else:
        st.error("Failed to initialize the chatbot. Please check your configuration.")

if __name__ == "__main__":
    main()
