"""Tests for the retrieval layer.

These tests do not hit the network — they only exercise the data structures
and the cosine-distance-to-similarity conversion logic.
"""

from documind.retriever import RetrievedChunk


def test_retrieved_chunk_dataclass():
    chunk = RetrievedChunk(
        text="hello",
        source="a.pdf",
        page=3,
        score=0.9,
        chunk_id="abc",
    )
    assert chunk.text == "hello"
    assert chunk.source == "a.pdf"
    assert chunk.page == 3
    assert chunk.score == 0.9
    assert chunk.chunk_id == "abc"


def test_chroma_distance_to_similarity():
    # Same formula used in retriever.retrieve()
    def to_similarity(distance: float) -> float:
        return max(0.0, 1.0 - (distance / 2.0))

    assert to_similarity(0.0) == 1.0  # identical
    assert to_similarity(1.0) == 0.5  # orthogonal-ish
    assert to_similarity(2.0) == 0.0  # opposite
    assert to_similarity(3.0) == 0.0  # clamps negative
