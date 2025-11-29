from sentence_transformers import SentenceTransformer
from typing import List
from .config import models

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print(f"[INFO] Loading embedding model: {models.embedding_model_name}")
        _model = SentenceTransformer(models.embedding_model_name)
    return _model

def embed_texts(texts: List[str]):
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return embeddings
