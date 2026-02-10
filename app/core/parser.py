import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


HEADING_VARIANTS = [
    r"projects?",
    r"academic projects",
    r"personal projects",
    r"experience",
    r"work experience",
    r"professional experience",
    r"education",
    r"academic background",
    r"skills",
    r"skills and competencies",
    r"technical skills",
    r"summary",
    r"objective",
]

HEADING_RE = re.compile(rf"^\s*(?P<h>{'|'.join(HEADING_VARIANTS)})\s*$", flags=re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*([\u2022\-\*]|\d+[\.)]|\([a-zA-Z0-9]\))\s+")


def normalize_section(heading_text: str) -> str:
    if not heading_text:
        return "unknown"
    h = heading_text.strip().lower()
    if "project" in h:
        return "projects"
    if "experience" in h or "work" in h or "professional" in h:
        return "experience"
    if "education" in h or "academic" in h:
        return "education"
    if "skill" in h or "competenc" in h:
        return "skills"
    if "summary" in h or "objective" in h:
        return "summary"
    return "other"


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split a full resume text into (heading_text, section_text) tuples.

    If no headings are found, returns a single section with heading_text="".
    """
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_heading = ""
    current_lines: List[str] = []

    for ln in lines:
        if HEADING_RE.match(ln):
            # flush previous
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = ln.strip()
            current_lines = []
        else:
            current_lines.append(ln)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    if not sections:
        return [("", text)]
    return sections


def split_section_into_items(section_text: str) -> List[str]:
    """Split a section into items: prefer bullets; otherwise split on blank lines.

    Keeps bullet groups intact (multiple wrapped lines for same bullet).
    """
    lines = section_text.splitlines()
    items: List[List[str]] = []
    i = 0
    # detect if there are any bullet markers in the section
    has_bullets = any(BULLET_RE.match(l) for l in lines if l.strip())

    if has_bullets:
        current: List[str] = []
        for ln in lines:
            if not ln.strip():
                # preserve blank lines as separators between bullets but don't create empty items
                continue
            if BULLET_RE.match(ln):
                if current:
                    items.append(current)
                current = [BULLET_RE.sub('', ln).rstrip()]
            else:
                # continuation line
                if current:
                    current.append(ln.rstrip())
                else:
                    # stray text (no bullet opened) -> start a pseudo-item
                    current = [ln.rstrip()]
        if current:
            items.append(current)
    else:
        # split by blank lines into paragraphs
        paragraph: List[str] = []
        for ln in lines:
            if ln.strip():
                paragraph.append(ln.rstrip())
            else:
                if paragraph:
                    items.append(paragraph)
                    paragraph = []
        if paragraph:
            items.append(paragraph)

    return ["\n".join(it).strip() for it in items if "\n".join(it).strip()]


def parse_text_to_documents(text: str, source: str = None, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """Parse resume text into Documents with section-aware metadata.

    - Detect headings and split into sections
    - Split sections into items (bullets/paragraphs)
    - Use RecursiveCharacterTextSplitter only within an item if it is very long
    """
    docs: List[Document] = []
    sections = split_into_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )

    for heading_text, section_text in sections:
        section = normalize_section(heading_text)
        if not heading_text:
            # heuristics for personal info: email, phone, or common profile links
            if re.search(r"\S+@\S+\.\S+", section_text) or re.search(r"\+?\d[\d\-\s\(\)]{7,}\d", section_text) or re.search(r"linkedin|github|portfolio", section_text, flags=re.IGNORECASE):
                section = "personal"

        items = split_section_into_items(section_text)

        for idx, item in enumerate(items):
            if not item:
                continue
            # if item is long, split further but remain in same section
            if len(item) > chunk_size:
                subdocs = splitter.split_documents([Document(page_content=item)])
                for sd_idx, sd in enumerate(subdocs):
                    md = dict(section=section, heading_text=heading_text or "", bullet_index=idx, subchunk_index=sd_idx)
                    if source:
                        md["source"] = source
                    sd.metadata.update(md) if getattr(sd, 'metadata', None) is not None else setattr(sd, 'metadata', md)
                    docs.append(sd)
            else:
                md = dict(section=section, heading_text=heading_text or "", bullet_index=idx)
                if source:
                    md["source"] = source
                docs.append(Document(page_content=item, metadata=md))

    return docs


__all__ = ["parse_text_to_documents", "split_into_sections", "split_section_into_items"]
