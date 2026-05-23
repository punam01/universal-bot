"""Lightweight evaluation harness.

Measures **retrieval quality** (source recall) of different pipeline
configurations against a saved golden set. Per-project storage.

LLM-as-judge for answer quality is intentionally NOT implemented yet —
it doubles cost and the simpler source-recall signal is already enough
to pick a good default. Can be added in a follow-up.

Golden set file format: one JSON object per line at
    {storage_dir}/eval_golden.jsonl              (default project)
    {storage_dir}/<project>/eval_golden.jsonl    (other projects)

Each line:
    {"question": str, "expected_sources": [str, ...], "notes": str?}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag import RAGEngine

from config import settings


@dataclass
class GoldenQuestion:
    question: str
    expected_sources: list[str]
    notes: str = ""


# ----- storage helpers -----

def _golden_path(project: str) -> Path:
    base = Path(settings.chroma_dir)
    if project == "default":
        return base / "eval_golden.jsonl"
    return base / project / "eval_golden.jsonl"


def load_golden(project: str) -> list[GoldenQuestion]:
    path = _golden_path(project)
    if not path.exists():
        return []
    out: list[GoldenQuestion] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                out.append(
                    GoldenQuestion(
                        question=data["question"],
                        expected_sources=list(data.get("expected_sources", [])),
                        notes=data.get("notes", ""),
                    )
                )
            except Exception:
                continue
    return out


def save_question(project: str, q: GoldenQuestion) -> None:
    path = _golden_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dedupe by question text — overwrite if already present.
    existing = [x for x in load_golden(project) if x.question != q.question]
    existing.append(q)
    with path.open("w") as f:
        for item in existing:
            f.write(json.dumps(asdict(item)) + "\n")


def delete_question(project: str, question_text: str) -> int:
    existing = load_golden(project)
    kept = [q for q in existing if q.question != question_text]
    removed = len(existing) - len(kept)
    if removed:
        path = _golden_path(project)
        with path.open("w") as f:
            for item in kept:
                f.write(json.dumps(asdict(item)) + "\n")
    return removed


# ----- metrics -----

def source_recall(retrieved: list[str], expected: list[str]) -> float:
    """Fraction of expected sources that appear in retrieved sources."""
    if not expected:
        return 1.0
    expected_set = set(expected)
    if not retrieved:
        return 0.0
    matched = sum(1 for s in expected_set if s in retrieved)
    return matched / len(expected_set)


# ----- config matrix -----

def all_common_configs(
    indexers: list[str],
    rewriters: list[str],
    rerankers: list[str],
) -> list[dict]:
    """Cartesian product of indexers x rewriters x rerankers, with 'none' on optional slots."""
    rewriters_with_none = ["none", *[r for r in rewriters if r != "none"]]
    rerankers_with_none = ["none", *[r for r in rerankers if r != "none"]]
    out: list[dict] = []
    for idx in indexers:
        for rw in rewriters_with_none:
            for rr in rerankers_with_none:
                out.append({"indexer": idx, "rewriter": rw, "reranker": rr})
    return out


# ----- runner -----

def _config_label(cfg: dict) -> str:
    return f"{cfg['indexer']} / {cfg['rewriter']} / {cfg['reranker']}"


def run_eval(
    engine: "RAGEngine",
    project: str,
    configs: list[dict],
    provider_key: str,
    top_k: int = 5,
) -> dict:
    """Run every config against every golden question. Return summary + details."""
    questions = load_golden(project)
    if not questions:
        return {
            "error": (
                "No golden questions yet. Ask a question in chat, then click "
                "'Save this Q to eval set' below the answer."
            )
        }

    per_config_recalls: dict[str, list[float]] = {}
    per_config_detail: dict[str, list[dict]] = {}

    for cfg in configs:
        label = _config_label(cfg)
        recalls: list[float] = []
        details: list[dict] = []
        for q in questions:
            try:
                sources = engine.retrieve(
                    cfg["indexer"],
                    q.question,
                    top_k=top_k,
                    reranker_key=cfg["reranker"],
                    rewriter_key=cfg["rewriter"],
                    provider_key=provider_key,
                )
                retrieved = [s["source"] for s in sources]
                recall = source_recall(retrieved, q.expected_sources)
                recalls.append(recall)
                details.append(
                    {
                        "question": q.question,
                        "expected": q.expected_sources,
                        "retrieved": retrieved,
                        "recall": recall,
                    }
                )
            except Exception as exc:
                recalls.append(0.0)
                details.append({"question": q.question, "error": str(exc)})
        per_config_recalls[label] = recalls
        per_config_detail[label] = details

    summary = []
    for label, recalls in per_config_recalls.items():
        avg = sum(recalls) / len(recalls) if recalls else 0.0
        summary.append(
            {
                "config": label,
                "avg_source_recall": round(avg, 3),
                "n_questions": len(recalls),
            }
        )
    summary.sort(key=lambda x: -x["avg_source_recall"])

    return {
        "summary": summary,
        "details": per_config_detail,
        "n_questions": len(questions),
        "n_configs": len(configs),
    }
