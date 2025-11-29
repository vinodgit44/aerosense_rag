from typing import List, Dict, Any
from .config import retrieval_cfg


def _extract_raw_text(doc: Any) -> str:
    """
    Accepts:
    - dict with 'text'
    - dict with any key
    - string
    Returns a standardized raw text string.
    """
    if isinstance(doc, str):
        return doc

    if isinstance(doc, dict):
        # common manual format: {"text": "...", ...}
        if "text" in doc:
            return doc["text"]

        # telemetry record: {"text": "...", "raw": {...}} etc.
        # fallback join of string values
        return " ".join(str(v) for v in doc.values())

    # fallback
    return str(doc)


def chunk_text(docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Break input documents into overlapping chunks.
    Supports:
        - list of strings
        - list of dicts containing 'text'
        - list of objects convertible to strings
    Returns list of chunks: [{"text": "...", "metadata": {...}}, ...]
    """
    chunks = []

    chunk_size = retrieval_cfg.chunk_size
    overlap = retrieval_cfg.chunk_overlap

    for doc in docs:
        raw_text = _extract_raw_text(doc)
        if not raw_text:
            continue

        # sliding window chunking
        start = 0
        while start < len(raw_text):
            end = start + chunk_size
            chunk = raw_text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": doc.get("source", "unknown") if isinstance(doc, dict) else "string_input"
                    }
                })

            start += chunk_size - overlap

    return chunks
