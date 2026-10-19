import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

DOCUMENT_PATH = Path(__file__).parent / "documents" / "financial_guide.txt"

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2"
)


def get_client():
    return genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            timeout=30000
        )
    )


def load_document():
    return DOCUMENT_PATH.read_text(encoding="utf-8")


def split_into_chunks(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embedding(text):
    client = get_client()

    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


def create_chunk_embeddings(chunks):
    client = get_client()

    contents = [
        types.Content(
            parts=[
                types.Part.from_text(text=chunk)
            ]
        )
        for chunk in chunks
    ]

    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=contents
    )

    chunk_embeddings = []

    for chunk, embedding in zip(chunks, response.embeddings):
        chunk_embeddings.append({
            "text": chunk,
            "embedding": embedding.values
        })

    return chunk_embeddings


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (magnitude_a * magnitude_b)


def find_relevant_chunks(question, chunk_embeddings, top_k=3):
    question_embedding = create_embedding(question)

    results = []

    for item in chunk_embeddings:
        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        results.append({
            "text": item["text"],
            "score": score
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]