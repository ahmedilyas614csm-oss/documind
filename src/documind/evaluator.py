"""RAG evaluation metrics: faithfulness, relevance, context precision.

Uses Groq as the LLM-as-judge. Metrics are inspired by RAGAS
(https://docs.ragas.io) but implemented from scratch so the code is
transparent, dependency-light, and easy to explain in a viva.

Design note: we ask the judge for a *single float* per metric rather
than a structured object. Smaller instruct models are far more reliable
at emitting one number than at emitting nested JSON.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

from documind.config import Settings, get_settings
from documind.ingest import _get_embedder


@dataclass
class EvalResult:
    """Container for the metrics on a single query."""

    query: str
    answer: str
    faithfulness: float  # 0..1, or -1 if judge failed
    answer_relevance: float  # 0..1
    context_precision: float  # 0..1
    n_contexts: int
    used_contexts: int
    judge_model: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _groq_client(settings: Settings):
    """Build a Groq-backed OpenAI-compatible client."""
    if not settings.has_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env to enable evaluation.")
    from openai import OpenAI

    return OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def _ask_judge(prompt: str, settings: Settings) -> str:
    """Send a prompt to the judge model and return the raw text response.

    Note: reasoning models (like gpt-oss) burn tokens thinking before they
    produce visible output, so we set max_tokens generously and do NOT
    force JSON mode — we just ask for a single number.
    """
    client = _groq_client(settings)

    resp = client.chat.completions.create(
        model=settings.groq_judge_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Reply with a single decimal number "
                    "between 0 and 1. Nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw = resp.choices[0].message.content
    if raw is None:
        return ""
    text = raw.strip()
    if os.getenv("DOCUMIND_DEBUG"):
        print(f"\n--- JUDGE RAW ---\n{text!r}\n-----------------\n")
    return text


def _extract_float(raw: str) -> float | None:
    """Pull the first float in [0, 1] out of a judge response."""
    if not raw:
        return None
    # Try JSON first: {"score": 0.8} or {"value": 0.8}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in ("score", "value", "result", "faithfulness", "precision"):
                if key in data:
                    return float(data[key])
            for v in data.values():
                if isinstance(v, (int, float)):
                    return float(v)
        if isinstance(data, (int, float)):
            return float(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback: first number like 0.83 or .83
    match = re.search(r"(0?\.\d+|[01](?:\.\d+)?)", raw)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            return None
    return None


# --- Metric 1: Faithfulness --------------------------------------------------


def faithfulness(answer: str, contexts: list[str], settings: Settings | None = None) -> float:
    """Fraction of the answer's claims supported by the contexts (0..1).

    Returns -1.0 if the judge response couldn't be interpreted.
    """
    settings = settings or get_settings()
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

    prompt = f"""CONTEXT:
{context_block}

ANSWER:
{answer}

Task: rate how faithful the ANSWER is to the CONTEXT on a scale of 0 to 1.
- 1.0 = every factual claim in the answer is directly supported by the context.
- 0.5 = about half the claims are supported.
- 0.0 = the answer contradicts or invents facts not in the context.

Reply with ONLY a number between 0 and 1."""

    raw = _ask_judge(prompt, settings)
    score = _extract_float(raw)
    return round(score, 4) if score is not None else -1.0


# --- Metric 2: Answer relevance ---------------------------------------------


def answer_relevance(question: str, answer: str, settings: Settings | None = None) -> float:
    """How well the answer addresses the question (0..1)."""
    settings = settings or get_settings()

    # Cheap signal: embedding cosine similarity
    embedder = _get_embedder()
    vecs = embedder.encode([question, answer], convert_to_numpy=True, show_progress_bar=False)
    q_vec, a_vec = vecs[0], vecs[1]
    denom = (float((q_vec * q_vec).sum()) ** 0.5) * (float((a_vec * a_vec).sum()) ** 0.5)
    cosine = float((q_vec * a_vec).sum()) / denom if denom else 0.0
    cosine = max(0.0, min(1.0, cosine))

    # Stronger signal: LLM judge
    prompt = f"""Question: {question}

Answer: {answer}

Task: rate how directly the Answer addresses the Question, on a scale of 0 to 1.
- 1.0 = the answer fully and directly addresses the question.
- 0.5 = partially addresses it or is vague.
- 0.0 = completely off-topic.

Reply with ONLY a number between 0 and 1."""

    raw = _ask_judge(prompt, settings)
    judge_score = _extract_float(raw)
    if judge_score is None:
        judge_score = cosine

    combined = 0.4 * cosine + 0.6 * judge_score
    return round(max(0.0, min(1.0, combined)), 4)


# --- Metric 3: Context precision --------------------------------------------


def context_precision(
    question: str, contexts: list[str], settings: Settings | None = None
) -> tuple[float, int]:
    """Fraction of retrieved chunks relevant to the question."""
    settings = settings or get_settings()
    if not contexts:
        return 0.0, 0

    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

    prompt = f"""Question: {question}

The following {len(contexts)} chunks were retrieved:
{context_block}

Task: rate what fraction of these chunks are actually relevant to answering
the question, on a scale of 0 to 1.
- 1.0 = every chunk is relevant.
- 0.5 = about half are relevant.
- 0.0 = none are relevant.

Reply with ONLY a number between 0 and 1."""

    raw = _ask_judge(prompt, settings)
    precision = _extract_float(raw)
    if precision is None:
        precision = 0.0

    used = round(precision * len(contexts))
    return round(precision, 4), used


# --- Aggregate --------------------------------------------------------------


def evaluate(
    query: str,
    answer: str,
    contexts: list[str],
    settings: Settings | None = None,
) -> EvalResult:
    """Run all metrics and return a single EvalResult."""
    settings = settings or get_settings()

    faith = faithfulness(answer, contexts, settings)
    relevance = answer_relevance(query, answer, settings)
    precision, n_used = context_precision(query, contexts, settings)

    return EvalResult(
        query=query,
        answer=answer,
        faithfulness=faith,
        answer_relevance=relevance,
        context_precision=precision,
        n_contexts=len(contexts),
        used_contexts=n_used,
        judge_model=settings.groq_judge_model,
        details={"context_lengths": [len(c) for c in contexts]},
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    from documind.generator import generate

    console = Console()
    q = "what is multi-head attention"
    console.print(f"[bold]Evaluating:[/bold] {q}\n")

    ans = generate(q)
    console.print(f"[dim]Answer preview:[/dim] {ans.text[:200]}...\n")

    result = evaluate(ans.query, ans.text, ans.contexts)
    table = Table(title="Evaluation metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_row(
        "Faithfulness",
        "judge failed" if result.faithfulness < 0 else f"{result.faithfulness:.4f}",
    )
    table.add_row("Answer relevance", f"{result.answer_relevance:.4f}")
    table.add_row("Context precision", f"{result.context_precision:.4f}")
    table.add_row("Contexts used", f"{result.used_contexts}/{result.n_contexts}")
    console.print(table)
