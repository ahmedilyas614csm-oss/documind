"""Vector retrieval: query → embed → nearest neighbors from Chroma.

Query embedding uses the same local sentence-transformers model
as ingestion, so no API call and no gRPC.
"""

from __future__ import annotations

from dataclasses import dataclass

from documind.config import Settings, get_settings
from documind.ingest import _get_embedder, get_collection


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score."""

    text: str
    source: str
    page: int
    score: float  # 0..1 — higher is more similar
    chunk_id: str


def _embed_query(query: str, settings: Settings) -> list[float]:
    """Embed a query with the same local model used during ingestion."""
    model = _get_embedder()
    vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    return vec[0].tolist()


def retrieve(
    query: str,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k most similar chunks for a query."""
    settings = settings or get_settings()
    top_k = top_k or settings.top_k

    collection = get_collection(settings)
    if collection.count() == 0:
        return []

    query_vec = _embed_query(query, settings)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns a list per query; we sent only one.
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for doc, meta, dist, cid in zip(docs, metas, dists, ids):
        # Chroma cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1]
        similarity = max(0.0, 1.0 - (dist / 2.0))
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=round(similarity, 4),
                chunk_id=cid,
            )
        )
    return chunks


if __name__ == "__main__":
    import sys

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    if len(sys.argv) < 2:
        console.print("Usage: python -m documind.retriever <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    hits = retrieve(query)
    if not hits:
        console.print("[yellow]No results — did you ingest any PDFs yet?[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold]Query:[/bold] {query}\n")
    for i, h in enumerate(hits, 1):
        console.print(
            Panel(
                h.text,
                title=f"[{i}] {h.source} p.{h.page}  ·  score={h.score}",
                border_style="cyan",
            )
        )
