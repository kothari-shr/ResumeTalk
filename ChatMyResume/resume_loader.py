from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # FIXED IMPORT
from langchain_core.documents import Document  # FIXED IMPORT

def load_and_split_resume(file_path: str) -> List[Document]:
    """
    Loads a PDF resume and splits it into semantically coherent chunks,
    preserving source/page metadata for later citation.
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Clean up: strip whitespace and normalize newlines
    cleaned = []
    for d in docs:
        text = " ".join(d.page_content.split())
        cleaned.append(Document(
            page_content=text,
            metadata=d.metadata
        ))

    # Resumes are short; smaller chunks with a bit of overlap work well
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n","•" ,"\n", ".", " ", ""],
    )
    split_docs = splitter.split_documents(cleaned)
    return split_docs