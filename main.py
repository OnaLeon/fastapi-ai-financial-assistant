import os
import time
import uuid

from rag import (
    load_document,
    split_into_chunks,
    create_chunk_embeddings,
    find_relevant_chunks,
)
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types, errors
from pydantic import BaseModel


# Load variables from .env
load_dotenv()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

app = FastAPI()

# Create Gemini client
_client = None

def get_client():
    global _client

    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=30000
            )
        )

    return _client
chunk_embeddings = None

def get_chunk_embeddings():
    global chunk_embeddings

    if chunk_embeddings is None:
        document = load_document()
        chunks = split_into_chunks(document)
        chunk_embeddings = create_chunk_embeddings(chunks)

    return chunk_embeddings


# Create one chat session
# This object will remember the conversation while the server is running
def create_chat():
    client = get_client()

    return client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction="""
            You are a financial education assistant.

            Answer questions using the provided CONTEXT.
            If the context does not contain enough information to answer the question,
            say that the information is not available in the provided document.

            Explain financial concepts clearly and simply.
            Keep answers concise.
            Do not give personalized financial advice.
            """
        )
    )

sessions = {}

# Models
class User(BaseModel):
    name: str
    age: int
    email: str


class Question(BaseModel):
    question: str


# Temporary users stored in memory
users = []


@app.get("/")
def home():
    return {
        "message": "Hello from my first FastAPI API"
    }


@app.get("/users")
def get_users():
    return users


@app.post("/users")
def create_user(user: User):
    users.append(user)
    return user

@app.post("/sessions")
def create_session():
    session_id = str(uuid.uuid4())

    sessions[session_id] = create_chat()

    return {
        "session_id": session_id
    }

@app.post("/sessions/{session_id}/ask")
def ask_question(session_id: str, request: Question):
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    chat = sessions[session_id]

    max_retries = 3

    for attempt in range(max_retries):
        try:
            relevant_chunks = find_relevant_chunks(
                request.question,
                get_chunk_embeddings(),
                top_k=3
            )
            context = "\n\n".join(
                item["text"] for item in relevant_chunks
            )
            prompt = f"""
            CONTEXT:
            {context}

            USER QUESTION:
            {request.question}
            """


            response = chat.send_message(
                message=prompt
            )

            return {
                "answer": response.text,
                "source": "financial_guide.txt",
                "retrieved_chunks": len(relevant_chunks)
            }

        except errors.APIError as e:
            print("Gemini API error:", e.code, e.message)

            if e.code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Gemini API quota has been reached. Please try again later."
                )

            if e.code == 503:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

                raise HTTPException(
                    status_code=503,
                    detail="AI service is temporarily unavailable."
                )

            if e.code == 404:
                raise HTTPException(
                    status_code=500,
                    detail="Configured Gemini model is not available."
                )

            raise HTTPException(
                status_code=502,
                detail="Unexpected error while communicating with the AI service."
            )

@app.post("/sessions/{session_id}/reset")
def reset_chat(session_id: str):
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    sessions[session_id] = create_chat()

    return {
        "message": "Conversation memory cleared."
    }

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    del sessions[session_id]

    return {
        "message": "Session deleted."
    }