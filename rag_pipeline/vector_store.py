import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

from .config import paths
from .embeddings import embed_texts

# Persistent database
_client = chromadb.PersistentClient(
    path=str(paths.vector_db_dir),
    settings=Settings(allow_reset=True)
)


def get_or_create_collection(name: str):
    try:
        return _client.get_collection(name=name)
    except Exception:
        return _client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )


def build_collection(
    name: str,
    docs: List[Dict[str, Any]],
    prefix: str,
    batch_size: int = 256
):
    """
    Build (or rebuild) a Chroma collection from a list of docs:
        docs: [{"text": "...", "metadata": {...}}, ...]
    """

    # Always delete old collection before indexing
    try:
        _client.delete_collection(name)
        print(f"[INFO] Removed existing collection '{name}' before rebuild.")
    except Exception:
        print(f"[INFO] No previous collection '{name}' found. Creating new one.")

    collection = _client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

    if not docs:
        print(f"[WARN] No docs passed to build_collection('{name}'). Skipping.")
        return collection

    ids = [f"{prefix}_{i}" for i in range(len(docs))]
    texts = [d.get("text", "") for d in docs]
    metadatas = [d.get("metadata", {"source": "unknown"}) for d in docs]

    # Filter out truly empty texts
    filtered_ids = []
    filtered_texts = []
    filtered_metadatas = []

    for i, t, m in zip(ids, texts, metadatas):
        if isinstance(t, str) and t.strip():
            filtered_ids.append(i)
            filtered_texts.append(t)
            filtered_metadatas.append(m)

    print(f"[INFO] Total non-empty docs to embed for '{name}': {len(filtered_texts)}")
    print(f"[INFO] Embedding in batches of {batch_size}")

    for start in range(0, len(filtered_texts), batch_size):
        end = start + batch_size
        batch_ids = filtered_ids[start:end]
        batch_texts = filtered_texts[start:end]
        batch_metadatas = filtered_metadatas[start:end]

        if not batch_texts:
            continue

        print(f"[INFO] Embedding batch {start} → {end} for '{name}'")

        batch_embeddings = embed_texts(batch_texts)

        # convert numpy array to list-of-lists if needed
        try:
            batch_embeddings = batch_embeddings.tolist()
        except AttributeError:
            pass

        # ensure metadata are dicts
        batch_metadatas = [
            m if isinstance(m, dict) and m else {"source": "unknown"}
            for m in batch_metadatas
        ]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )

    print(f"[SUCCESS] Collection '{name}' built with {collection.count()} items.")
    return collection


def query_collection(name: str, query: str, n_results: int = 5):
    """
    Query a Chroma collection using text similarity.
    Returns:
        {
            "ids": [...],
            "documents": [...],
            "metadatas": [...],
            "distances": [...]
        }
    """
    try:
        collection = _client.get_collection(name=name)
    except Exception as e:
        print(f"[ERROR] Cannot load collection '{name}': {e}")
        return None

    from .embeddings import embed_texts
    query_emb = embed_texts([query])[0]  # numpy vector

    try:
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n_results,
        )
        return results
    except Exception as e:
        print(f"[ERROR] Query failed on '{name}': {e}")
        return None
