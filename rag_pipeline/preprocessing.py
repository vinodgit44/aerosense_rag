import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - remove newlines
    - collapse multiple spaces
    - trim whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
