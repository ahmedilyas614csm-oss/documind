"""High-level orchestration: one entry point for the whole RAG pipeline.

The Streamlit app and CLI both call into this module. Keeping the wiring
here (and not in the UI) means the same pipeline can be driven from a
notebook, a script, or a web service without changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Quiet the HuggingFace Hub warnings before any HF import happens.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console

from documind.config import Settings, get_settings
from documind.evaluator import EvalResult, evaluate
from documind.generator import Answer, generate
from documind.ingest import get_collection, ingest_directory, ingest_pdf
from documind.retriever import retrieve

console = Console()


@dataclass
class QueryResult:
    """Everything the UI needs to render one Q&A turn."""

    answer: Answer
    evaluation: EvalResult
    contexts_used: int
    contexts_total: int


def ingest_path(path: str | Path, settings: Settings | None = None) -> int:
    """Ingest a single PDF or every PDF in a directory. Returns chunks stored."""
    settings = settings or get_settings()
    target = Path(path)

    if target.is_dir():
        return ingest_directory(target, settings)
    if target.suffix.lower() == ".pdf":
        return ingest_pdf(target, settings)
    raise ValueError(f"Unsupported path: {target}. Expected a .pdf or a directory.")


def collection_size(settings: Settings | None = None) -> int:
    """Return how many chunks are currently indexed."""
    settings = settings or get_settings()
    return get_collection(settings).count()


def run_query(
    query: str,
    top_k: int | None = None,
    with_evaluation: bool = True,
    settings: Settings | None = None,
) -> QueryResult:
    """Retrieve, generate, and (optionally) evaluate an answer for a query."""
    settings = settings or get_settings()

    answer = generate(query, top_k=top_k, settings=settings)

    if with_evaluation and answer.contexts:
        evaluation = evaluate(
            query=query,
            answer=answer.text,
            contexts=answer.contexts,
            settings=settings,
        )
    else:
        # Skip evaluation — return a placeholder so the dataclass shape holds.
        from documind.evaluator import EvalResult

        evaluation = EvalResult(
            query=query,
            answer=answer.text,
            faithfulness=float("nan"),
            answer_relevance=float("nan"),
            context_precision=float("nan"),
            n_contexts=len(answer.contexts),
            used_contexts=0,
            judge_model="(skipped)",
        )

    return QueryResult(
        answer=answer,
        evaluation=evaluation,
        contexts_used=evaluation.used_contexts,
        contexts_total=len(answer.contexts),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("Usage: python -m documind.pipeline <question>")
        console.print("   or: python -m documind.pipeline --ingest <path>")
        sys.exit(1)

    if sys.argv[1] == "--ingest":
        if len(sys.argv) < 3:
            console.print("[red]Need a path: --ingest <pdf_or_dir>[/red]")
            sys.exit(1)
        n = ingest_path(sys.argv[2])
        console.print(f"[bold green]Indexed {n} chunks[/bold green]")
        sys.exit(0)

    question = " ".join(sys.argv[1:])
    console.print(f"\n[bold]Q:[/bold] {question}")
    console.print(f"[dim]Indexed chunks: {collection_size()}[/dim]\n")

    result = run_query(question)
    console.print(f"[green]{result.answer.text}[/green]\n")
    console.print("[bold]Metrics:[/bold]")
    console.print(f"  Faithfulness:      {result.evaluation.faithfulness:.4f}")
    console.print(f"  Answer relevance:  {result.evaluation.answer_relevance:.4f}")
    console.print(f"  Context precision: {result.evaluation.context_precision:.4f}")
    console.print(f"  Contexts used:     {result.contexts_used}/{result.contexts_total}")
