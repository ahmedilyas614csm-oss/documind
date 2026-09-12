"""PDF ingestion: load → chunk → embed → store in Chroma.

Embeddings run locally via sentence-transformers (no API, no gRPC,
no segfaults). Gemini is only used later for answer generation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rich.console import Console

from documind.config import Settings, get_settings

console = Console()


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    source: str  # file name
    page: int  # 1-indexed page number
    chunk_id: str  # deterministic hash


def _hash(text: str) -> str:
    """Stable chunk ID from the content (avoids duplicate inserts)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, page_text), ...] for a PDF."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split pages into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 20:  # skip tiny fragments
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    page=page_num,
                    chunk_id=_hash(f"{source}:{page_num}:{piece}"),
                )
            )
    return chunks


# --- Embedding (local, no API) ----------------------------------------------

_EMBEDDER = None


def _get_embedder():
    """Lazy-load a local sentence-transformers model (cached per process)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        console.print("[dim]Loading embedding model (first time ~90MB)...[/dim]")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed texts locally using sentence-transformers. No API, no crashes."""
    model = _get_embedder()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


# --- Vector store ------------------------------------------------------------


def get_chroma_client(settings: Settings) -> chromadb.PersistentClient:
    """Return a persistent Chroma client."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(settings: Settings | None = None):
    """Return (creating if needed) the Chroma collection."""
    settings = settings or get_settings()
    client = get_chroma_client(settings)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# --- Ingest ------------------------------------------------------------------


def ingest_pdf(path: Path, settings: Settings | None = None) -> int:
    """Ingest one PDF. Returns number of chunks stored."""
    settings = settings or get_settings()
    collection = get_collection(settings)

    console.print(f"[cyan]Reading[/cyan] {path.name}")
    pages = load_pdf(path)
    chunks = chunk_pages(pages, path.name, settings.chunk_size, settings.chunk_overlap)

    if not chunks:
        console.print(f"[yellow]No extractable text in {path.name}[/yellow]")
        return 0

    console.print(f"[cyan]Embedding[/cyan] {len(chunks)} chunks from {path.name}")
    embeddings = _embed_texts([c.text for c in chunks], settings)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )
    console.print(f"[green]Stored {len(chunks)} chunks from {path.name}[/green]")
    return len(chunks)


def ingest_directory(directory: Path, settings: Settings | None = None) -> int:
    """Ingest every PDF in a directory. Returns total chunks stored."""
    settings = settings or get_settings()
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {directory}[/yellow]")
        return 0

    total = 0
    for pdf in pdfs:
        total += ingest_pdf(pdf, settings)
    console.print(f"[bold green]Total: {total} chunks stored[/bold green]")
    return total


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("Usage: python -m documind.ingest <pdf_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        ingest_directory(target)
    else:
        ingest_pdf(target)
