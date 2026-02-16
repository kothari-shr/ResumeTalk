import os
from typing import Optional
import logging

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Optional local embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# CORRECT IMPORTS for LangChain 1.0+
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import settings

logger = logging.getLogger(__name__)

def _get_embeddings():
    if settings.use_local_embeddings:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return OpenAIEmbeddings(model=settings.embeddings_model)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)


def build_or_load_vectorstore(docs, embeddings, index_path: Optional[str]) -> FAISS:
    """
    Build FAISS from docs or load from disk if present. Saves reprocessing of document everytime
    """
    if index_path and os.path.isdir(index_path):
        if settings.allow_dangerous_deserialization:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.warning("Vectorstore exists but deserialization is disabled; rebuilding index.")

    vs = FAISS.from_documents(docs, embeddings)
    if index_path:
        os.makedirs(index_path, exist_ok=True)
        vs.save_local(index_path)
    return vs


def format_docs(docs):
    """Format retrieved documents into a single string."""
    parts = [doc.page_content.strip() for doc in docs]
    # separate chunks clearly so the model understands chunk boundaries
    return "\n\n---\n\n".join(parts)


def _doc_matches_filters(doc, filters: dict) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    for k, v in filters.items():
        if k not in meta:
            return False
        mv = str(meta.get(k, "")).lower()
        if isinstance(v, (list, tuple, set)):
            values = [str(item).lower() for item in v]
            if not any(val in mv for val in values):
                return False
        elif isinstance(v, str):
            if v.lower() not in mv:
                return False
        else:
            if str(v).lower() != mv:
                return False
    return True


def _filter_documents_by_metadata(docs, filters: dict):
    """Return docs that match all key/value pairs in filters by metadata.

    If no filters provided, return docs unchanged. If filtering leaves no
    documents, return the original list (avoid empty context to the LLM).
    """
    if not filters:
        return docs

    filtered = [d for d in docs if _doc_matches_filters(d, filters)]
    if not filtered:
        # fallback to original if overly restrictive
        return docs
    return filtered


def _get_docs_from_vectorstore(vectorstore, filters: dict | None = None):
    """Return all docs from the vectorstore that match filters.

    If filters are omitted or empty, returns all docs in the store.
    """
    if not vectorstore:
        return []

    docstore = getattr(vectorstore, "docstore", None)
    store_dict = getattr(docstore, "_dict", None)
    if not store_dict:
        return []

    docs = list(store_dict.values())
    if filters:
        filtered = [d for d in docs if d and _doc_matches_filters(d, filters)]
        if not filtered:
            return []
    else:
        filtered = [d for d in docs if d]

    def sort_key(doc):
        meta = getattr(doc, "metadata", {}) or {}
        return (
            meta.get("section", ""),
            int(meta.get("bullet_index", 0) or 0),
            int(meta.get("subchunk_index", 0) or 0),
        )

    return sorted(filtered, key=sort_key)


def _limit_docs_for_context(docs):
    """Limit docs by max count and max total characters to avoid prompt overflow."""
    max_docs = settings.max_context_docs
    max_chars = settings.max_context_chars

    if max_docs <= 0 and max_chars <= 0:
        return docs

    limited = []
    total_chars = 0
    for doc in docs:
        content = (doc.page_content or "").strip()
        content_len = len(content)

        if max_docs > 0 and len(limited) >= max_docs:
            break
        if max_chars > 0 and total_chars + content_len > max_chars:
            break

        limited.append(doc)
        total_chars += content_len

    return limited


def build_conv_rag_chain(vectorstore):
    """
    Build a Conversational RAG chain using pure LCEL (LangChain Expression Language).
    This works with LangChain 1.0+
    """
    llm = _get_llm()
    
    # Prompt to reformulate question based on chat history
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question. If it's already standalone, return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    # QA prompt — detailed, context-rich, references specific experience, roles, and projects
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an AI assistant helping with technical interviews. 
Given the following context from the candidate's resume and previous chat history, answer the interviewer's question in a way that:
- Clearly states whether the candidate has experience with the mentioned skill.
- References specific roles, projects, or achievements from the context (such as job titles, personal projects, or industry experience or company name) if he has any. 
Even if he does not have direct experience, mention related skills or experiences or how candidate could potentially apply their knowledge.
- Is detailed with proper format which makes it easy and readable for the interviewer and provides enough detail to impress an interviewer. If required also provide the personal projects (and github link if present) or course work that are relevant.
- If the answer is not present in the resume context, respond exactly: \"I don't know.\"
- Keep the conversation engaging which could lead interviwer to ask to connect with the candidate. The personal information should be retrived from the resume only. 
Context from resume:
{context}
""",
        ),
        ("human", "{question}"),
    ])
    
    # Create the full chain
    chain = (
        RunnablePassthrough.assign(
            standalone_question=lambda x: (
                (condense_question_prompt | llm | StrOutputParser()).invoke(x)
                if x.get("chat_history") else x["question"]
            )
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(
                _limit_docs_for_context(
                    _get_docs_from_vectorstore(vectorstore, x.get("filters", {}) or None)
                )
            )
        )
        | RunnablePassthrough.assign(
            answer=lambda x: (qa_prompt | llm | StrOutputParser()).invoke({
                "question": x["standalone_question"],
                "context": x["context"]
            })
        )
    )
    
    return chain


def bootstrap_rag(docs):
    """
    Compose embeddings, vector store, and chain.
    """
    embeddings = _get_embeddings()
    index_path = settings.vectorstore_path or None
    vs = build_or_load_vectorstore(docs, embeddings, index_path)
    chain = build_conv_rag_chain(vs)
    return chain