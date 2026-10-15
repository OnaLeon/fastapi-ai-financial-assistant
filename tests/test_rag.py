from rag import split_into_chunks, cosine_similarity


def test_split_into_chunks():
    text = "one two three four five six"

    chunks = split_into_chunks(
        text,
        chunk_size=4,
        overlap=1
    )

    assert len(chunks) == 2


def test_identical_vectors_have_similarity_one():
    vector = [1.0, 2.0, 3.0]

    score = cosine_similarity(
        vector,
        vector
    )

    assert score == 1.0


def test_different_vectors_have_lower_similarity():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]

    score = cosine_similarity(
        vector_a,
        vector_b
    )

    assert score == 0.0