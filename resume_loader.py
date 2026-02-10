from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from app.core.parser import parse_text_to_documents


def load_and_split_resume(file_path: str) -> List[Document]:
    """
    Loads a PDF resume and performs section-aware parsing that preserves
    bullet structure and tags each chunk with metadata: section, heading_text, bullet_index.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    # Combine pages with a page separator to retain some structure
    page_texts = []
    for i, p in enumerate(pages):
        # normalize whitespace within page
        text = "\n".join(line.rstrip() for line in p.page_content.splitlines())
        page_texts.append(text)

    full_text = "\n\n---PAGE_BREAK---\n\n".join(page_texts)

    # parse into Documents with section-aware metadata
    docs = parse_text_to_documents(full_text, source=file_path)
    return docs