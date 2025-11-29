from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

from .retrieval import retrieve_uav_docs


@dataclass
class EvalSample:
    query: str
    expected_ids: List[str]     # IDs that should appear
    description: str = ""


@dataclass
class EvalResult:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hits: int
    total_expected: int


def compute_precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for item in retrieved_k if item in expected_ids)
    return hits / k


def compute_recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for item in retrieved_k if item in expected_ids)
    return hits / len(expected_ids) if expected_ids else 0.0


def compute_mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def run_single_eval(sample: EvalSample, k: int = 5) -> EvalResult:
    retrieved = retrieve_uav_docs(sample.query, top_k_manual=k, top_k_telemetry=k)
    retrieved_ids = [
        doc.metadata.get("source", "") + "_" + doc.source_type
        for doc in retrieved
    ]

    precision = compute_precision_at_k(retrieved_ids, sample.expected_ids, k)
    recall = compute_recall_at_k(retrieved_ids, sample.expected_ids, k)
    mrr = compute_mrr(retrieved_ids, sample.expected_ids)

    hits = sum(1 for rid in retrieved_ids if rid in sample.expected_ids)

    return EvalResult(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        hits=hits,
        total_expected=len(sample.expected_ids)
    )


def run_eval_suite(samples: List[EvalSample], k: int = 5) -> Dict[str, Any]:
    results = []
    for s in samples:
        res = run_single_eval(s, k=k)
        results.append(res)

    precision_list = [r.precision_at_k for r in results]
    recall_list = [r.recall_at_k for r in results]
    mrr_list = [r.mrr for r in results]

    return {
        "precision@k": float(np.mean(precision_list)),
        "recall@k": float(np.mean(recall_list)),
        "MRR": float(np.mean(mrr_list)),
        "samples": results,
    }
