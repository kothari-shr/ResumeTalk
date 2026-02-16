import re
from typing import Dict, Any

from langchain_community.document_loaders import PyMuPDFLoader


_SECTION_KEYWORDS = {
    "projects": ["project", "projects", "personal projects", "academic projects"],
    "experience": ["experience", "work", "professional", "roles"],
    "education": ["education", "degree", "university", "school", "bachelor", "master", "phd"],
    "skills": ["skill", "skills", "competence", "technology", "stack"],
}

_SECTION_PATTERNS = {
    section: re.compile(r"\b(?:" + "|".join(map(re.escape, kws)) + r")\b", re.IGNORECASE)
    for section, kws in _SECTION_KEYWORDS.items()
}


def _match_sections_by_keywords(query: str) -> list[str]:
    matched = [section for section, pat in _SECTION_PATTERNS.items() if pat.search(query)]
    return sorted(set(matched))


def classify_intent(query: str) -> Dict[str, Any]:
    """Classify a user query into target sections and return context filters.

    Returns a dict with:
    - target_sections: list of sections (e.g. ['projects', 'experience'])
    - normalized_filters: dict suitable to filter context (e.g. {'section': ['projects']})

    This implementation uses keyword matching and normalizes common variants
    (e.g., 'project', 'projects' -> 'projects'). It intentionally avoids few-shot
    hard-coded examples and does not add exclusion filters.
    """
    if not query or not query.strip():
        return {"target_sections": [], "normalized_filters": {}}

    targets = _match_sections_by_keywords(query)

    filters: Dict[str, Any] = {}
    if targets:
        filters["section"] = targets

    return {"target_sections": targets, "normalized_filters": filters}


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
