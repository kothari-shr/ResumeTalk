import os
from typing import Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Optional local embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# CORRECT IMPORTS for LangChain 1.0+
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.messages import HumanMessage, AIMessage


def _get_embeddings():
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    if use_local:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)


def _get_llm() -> ChatOpenAI:
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=model, temperature=temperature)


def build_or_load_vectorstore(docs, embeddings, index_path: Optional[str]) -> FAISS:
    """
    Build FAISS from docs or load from disk if present. Saves reprocessing of document everytime
    """
    if index_path and os.path.isdir(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    vs = FAISS.from_documents(docs, embeddings)
    if index_path:
        os.makedirs(index_path, exist_ok=True)
        vs.save_local(index_path)
    return vs


def get_retriever(vs: FAISS):
    """
    Use MMR to diversify retrieved chunks.
    """
    k = int(os.getenv("RETRIEVER_K", "4"))
    fetch_k = int(os.getenv("RETRIEVER_FETCH_K", "12"))
    lambda_mult = float(os.getenv("MMR_LAMBDA", "0.7"))
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_conv_rag_chain(retriever):
    """
    Build a Conversational RAG chain using pure LCEL (LangChain Expression Language).
    This works with LangChain 1.0+
    """
    llm = _get_llm()
    
    # Prompt to reformulate question based on chat history
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question. If it's already standalone, return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an interview assistant that answers ONLY using the candidate's resume content.
            If the answer is not present in the resume context, say "I don't know."
            Keep answers concise and helpful for interviews. Use bullet points when listing items.
            Cite pages if relevant.

        Context from resume:
        {context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
    
    # Build the chain using LCEL
    def condense_question(inputs):
        """Condense the question if there's chat history, otherwise pass through."""
        if inputs.get("chat_history"):
            return condense_question_prompt | llm | StrOutputParser()
        else:
            return RunnablePassthrough()
    
    # Create the full chain
    chain = (
        RunnablePassthrough.assign(
            standalone_question=lambda x: (
                (condense_question_prompt | llm | StrOutputParser()).invoke(x)
                if x.get("chat_history") else x["input"]
            )
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        )
        | RunnablePassthrough.assign(
            answer=(qa_prompt | llm | StrOutputParser())
        )
    )
    
    return chain


def bootstrap_rag(docs) -> Tuple:
    """
    Compose embeddings, vector store, retriever, and chain.
    """
    embeddings = _get_embeddings()
    index_path = os.getenv("VECTORSTORE_PATH", ".faiss_index") or None
    vs = build_or_load_vectorstore(docs, embeddings, index_path)
    retriever = get_retriever(vs)
    chain = build_conv_rag_chain(retriever)
    return chain, vs