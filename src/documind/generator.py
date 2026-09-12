"""Answer generation using Groq's LLM API.

Takes retrieved chunks + a query and produces a grounded, cited answer.
Uses the OpenAI Python SDK pointed at Groq's endpoint — no gRPC, no crashes.
"""

from __future__ import annotations

from dataclasses import dataclass

from documind.config import Settings, get_settings
from documind.retriever import RetrievedChunk, retrieve


@dataclass
class Answer:
    """A generated answer with its source citations."""

    text: str
    citations: list[dict]  # [{source, page, chunk_id, score}, ...]
    model: str
    query: str
    contexts: list[str]  # raw retrieved chunk texts (for evaluation)


SYSTEM_PROMPT = """You are a helpful research assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say "I don't have enough information to answer that."
Always cite your sources by referencing the source number, e.g. [Source 1].

Be concise and accurate. Do not make up information."""


def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Format the retrieved chunks + query into a single prompt."""
    parts = [f"Question: {query}\n\nContext:"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"\n[Source {i}] ({c.source}, page {c.page})\n{c.text}")
    parts.append("\n\nAnswer the question using only the context above.")
    return "\n".join(parts)


def generate(
    query: str,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> Answer:
    """Retrieve relevant chunks and generate a grounded answer via Groq."""
    settings = settings or get_settings()

    # 1. Retrieve
    chunks = retrieve(query, top_k=top_k, settings=settings)
    if not chunks:
        return Answer(
            text="I don't have enough information to answer that. (No documents indexed.)",
            citations=[],
            model=settings.groq_model,
            query=query,
            contexts=[],
        )

    # 2. Call Groq
    if not settings.has_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env to enable answer generation.")

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, chunks)},
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.max_answer_tokens,
    )

    answer_text = response.choices[0].message.content.strip()

    citations = [
        {
            "source": c.source,
            "page": c.page,
            "chunk_id": c.chunk_id,
            "score": c.score,
        }
        for c in chunks
    ]

    return Answer(
        text=answer_text,
        citations=citations,
        model=settings.groq_model,
        query=query,
        contexts=[c.text for c in chunks],
    )


if __name__ == "__main__":
    import sys

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    if len(sys.argv) < 2:
        console.print("Usage: python -m documind.generator <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    console.print(f"\n[bold]Q:[/bold] {question}\n")

    result = generate(question)
    console.print(Panel(Markdown(result.text), title="Answer", border_style="green"))
    console.print("\n[bold]Citations:[/bold]")
    for i, c in enumerate(result.citations, 1):
        console.print(f"  [{i}] {c['source']} p.{c['page']}  (score {c['score']})")
