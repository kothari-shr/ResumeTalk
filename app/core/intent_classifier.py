import re
from typing import Dict, Any

from langchain_community.document_loaders import PyMuPDFLoader


_SECTION_KEYWORDS = {
    "projects": [r"project", r"projects", "personal projects", "academic projects"],
    "experience": [r"experience", r"work", r"professional", r"roles"],
    "education": [r"education", r"degree", r"university", r"school", r"bachelor", r"master", r"phd"],
    "skills": [r"skill", r"skills", r"competence", r"technology", r"stack"],
}


def _match_section_by_keywords(query: str) -> str:
    q = query.lower()
    # explicit multi-word rules first
    if "personal projects" in q or "academic projects" in q:
        return "projects"

    for section, kws in _SECTION_KEYWORDS.items():
        for kw in kws:
            if re.search(r"\b" + kw + r"\b", q):
                return section
    return "general"


def classify_intent(query: str) -> Dict[str, Any]:
    """Classify a user query into a target section and return retriever filters.

    Returns a dict with:
      - target_section: one of {projects, experience, education, skills, general}
      - normalized_filters: dict suitable to pass to the retriever (e.g. {'section': 'projects'})

    This implementation uses keyword matching and normalizes common variants
    (e.g., 'project', 'projects' -> 'projects'). It intentionally avoids few-shot
    hard-coded examples and does not add exclusion filters.
    """
    if not query or not query.strip():
        return {"target_section": "general", "normalized_filters": {}}

    target = _match_section_by_keywords(query)

    filters: Dict[str, Any] = {}
    if target and target != "general":
        filters["section"] = target

    return {"target_section": target, "normalized_filters": filters}


def detect_bullet_styles_from_text(text: str) -> Dict[str, int]:
    """Detect common bullet markers in a text block. Returns counts per style.

    Looks for: '•' (U+2022), '-', '*', numbered bullets (1., 1), '(a)', etc.
    """
    patterns = {
        "bullet_bullet": re.compile(r"^\s*[\u2022\u2023\u25E6]\s+", re.MULTILINE),
        "dash": re.compile(r"^\s*-\s+", re.MULTILINE),
        "star": re.compile(r"^\s*\*\s+", re.MULTILINE),
        "numbered": re.compile(r"^\s*\d+[\.)]\s+", re.MULTILINE),
        "paren_letter": re.compile(r"^\s*\([a-zA-Z0-9]+\)\s+", re.MULTILINE),
    }
    counts = {}
    for name, pat in patterns.items():
        counts[name] = len(pat.findall(text))
    return counts

def detect_bullet_styles_from_pdf(file_path: str) -> Dict[str, int]:
    """Load a PDF via PyMuPDFLoader and detect bullet styles across pages."""
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    combined = "\n\n".join(p.page_content for p in pages)
    return detect_bullet_styles_from_text(combined)


__all__ = ["classify_intent", "detect_bullet_styles_from_text", "detect_bullet_styles_from_pdf"]
