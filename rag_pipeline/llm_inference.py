from typing import List
import requests

from .config import models
from .retrieval import RetrievedDoc

OLLAMA_URL = "http://localhost:11434/api/generate"


SYSTEM_PROMPT = (
    "You are a UAV systems troubleshooting assistant. "
    "You get two kinds of context:\n"
    "- Engineering manual snippets\n"
    "- Telemetry summaries (sensor logs)\n\n"
    "Your job:\n"
    "- Propose likely root causes\n"
    "- Suggest what to inspect or test\n"
    "- Mention both manual procedures and data patterns when relevant\n"
    "- Be honest if context is insufficient.\n"
)


def build_rag_prompt(query: str, retrieved_docs: List[RetrievedDoc]) -> str:
    """
    Build a single text prompt for Ollama-style chat/generate endpoint.
    """

    context_blocks = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        src = doc.metadata.get("source", "")
        ts = doc.metadata.get("timestamp", "")
        header = f"[{idx} | {doc.source_type.upper()} | source={src} | timestamp={ts}]"
        context_blocks.append(header + "\n" + doc.text)

    context_text = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User issue:\n{query}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Now, based only on this context, provide:\n"
        "- likely root causes\n"
        "- key signals or patterns to confirm\n"
        "- recommended checks and corrective actions.\n\n"
        "Answer:\n"
    )
    return prompt


def call_ollama(prompt: str, model_name: str | None = None, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Calls local Ollama server's /api/generate endpoint.
    Assumes Ollama is running on default port (11434).
    """

    if model_name is None:
        model_name = models.ollama_model

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        return "LLM backend (Ollama) error: could not generate response."

    data = resp.json()
    return data.get("response", "").strip()


def generate_answer(query: str, retrieved_docs: List[RetrievedDoc], temperature: float = 0.2) -> str:
    """
    High-level helper:
    - build prompt
    - call Ollama
    """
    if not retrieved_docs:
        return "No relevant context retrieved. Please check your data/index."

    prompt = build_rag_prompt(query, retrieved_docs)
    answer = call_ollama(prompt, temperature=temperature)
    return answer
