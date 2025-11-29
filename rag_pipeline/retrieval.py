from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .vector_store import query_collection
from .config import retrieval_cfg


@dataclass
class RetrievedDoc:
    text: str
    metadata: Dict[str, Any]
    distance: float
    source_type: str   # "manual" or "telemetry"
    score: float       # fused score (normalized similarity * weight)


def _normalize_distances(distances: List[float]) -> List[float]:
    """
    Chroma returns distances where smaller = closer.
    We'll convert to similarity in [0, 1] by simple min-max scaling and inverting.
    """
    if not distances:
        return []
    mn = min(distances)
    mx = max(distances)
    if mx == mn:
        # all equal -> treat as 1.0 similarity
        return [1.0 for _ in distances]
    return [1.0 - (d - mn) / (mx - mn) for d in distances]


def _extract_results(results: Dict[str, Any], source_type: str, weight: float) -> List[RetrievedDoc]:
    if not results or not results.get("ids"):
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    sims = _normalize_distances(dists)

    retrieved: List[RetrievedDoc] = []
    for text, meta, dist, sim in zip(docs, metas, dists, sims):
        # ensure metadata is at least a dict
        if not isinstance(meta, dict):
            meta = {}
        retrieved.append(
            RetrievedDoc(
                text=text,
                metadata=meta,
                distance=dist,
                source_type=source_type,
                score=sim * weight,
            )
        )
    return retrieved


def retrieve_uav_docs(
    query: str,
    top_k_manual: Optional[int] = None,
    top_k_telemetry: Optional[int] = None,
) -> List[RetrievedDoc]:
    """
    Main retrieval entry point for AeroSense RAG.

    - queries manuals and telemetry independently
    - normalizes their scores
    - fuses them with weights
    - returns globally ranked top_k
    """

    if top_k_manual is None:
        top_k_manual = retrieval_cfg.top_k
    if top_k_telemetry is None:
        top_k_telemetry = retrieval_cfg.top_k

    # 1) manual retrieval
    try:
        manual_res = query_collection("manual_chunks", query, top_k_manual)
    except Exception as e:
        print(f"[WARN] manual_chunks retrieval failed: {e}")
        manual_res = None

    manual_docs = _extract_results(
        manual_res,
        source_type="manual",
        weight=retrieval_cfg.manual_weight,
    )

    # 2) telemetry retrieval
    try:
        telem_res = query_collection("telemetry_records", query, top_k_telemetry)
    except Exception as e:
        print(f"[WARN] telemetry_records retrieval failed: {e}")
        telem_res = None

    telem_docs = _extract_results(
        telem_res,
        source_type="telemetry",
        weight=retrieval_cfg.log_weight,
    )

    # 3) fuse & sort
    all_docs: List[RetrievedDoc] = manual_docs + telem_docs
    all_docs.sort(key=lambda d: d.score, reverse=True)

    # 4) keep only global top_k
    return all_docs[: retrieval_cfg.top_k]
