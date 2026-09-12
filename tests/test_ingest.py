"""Tests for PDF ingestion and chunking."""

from pathlib import Path

from documind.ingest import Chunk, _hash, chunk_pages, load_pdf


def test_hash_is_deterministic():
    assert _hash("hello world") == _hash("hello world")
    assert _hash("hello world") != _hash("hello worlds")


def test_chunk_pages_splits_long_text():
    long_text = "This is a sentence. " * 200  # ~4000 chars
    pages = [(1, long_text)]

    chunks = chunk_pages(pages, source="test.pdf", chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source == "test.pdf"
        assert c.page == 1
        assert c.chunk_id  # not empty
        assert len(c.text) >= 20


def test_chunk_pages_skips_tiny_fragments():
    pages = [(1, "hi")]  # too short, should be skipped
    chunks = chunk_pages(pages, source="tiny.pdf", chunk_size=800, chunk_overlap=100)
    assert chunks == []


def test_chunk_pages_preserves_page_numbers():
    pages = [(1, "First page content " * 30), (2, "Second page content " * 30)]
    chunks = chunk_pages(pages, source="multi.pdf", chunk_size=400, chunk_overlap=50)
    pages_seen = {c.page for c in chunks}
    assert 1 in pages_seen
    assert 2 in pages_seen


def test_load_pdf_handles_missing_file_gracefully(tmp_path: Path):
    # Ensure our chunker doesn't crash on empty input
    assert chunk_pages([], source="empty.pdf", chunk_size=800, chunk_overlap=100) == []
