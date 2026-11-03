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
from fastapi import Depends, FastAPI, HTTPException
from google import genai
from google.genai import types, errors
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database import get_db
from models import ConversationSession, Message

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
def create_chat(history=None):
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
        ),
        history=history or []
    )

def get_or_restore_chat(session_id: str, db: Session):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    db_session = db.get(
        ConversationSession,
        session_uuid
    )

    if db_session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    if session_id in sessions:
        return sessions[session_id]

    db_messages = db.scalars(
        select(Message)
        .where(Message.session_id == session_uuid)
        .order_by(Message.id)
    ).all()

    history = []

    for message in db_messages:
        role = (
            "model"
            if message.role == "assistant"
            else "user"
        )

        history.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(
                        text=message.content
                    )
                ]
            )
        )

    chat = create_chat(history=history)

    sessions[session_id] = chat

    return chat

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
def create_session(db: Session = Depends(get_db)):
    db_session = ConversationSession()

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    session_id = str(db_session.id)

    sessions[session_id] = create_chat()

    return {
        "session_id": session_id
    }

@app.post("/sessions/{session_id}/ask")
def ask_question(
    session_id: str,
    request: Question,
    db: Session = Depends(get_db)
):

    chat = get_or_restore_chat(
        session_id,
        db
    )
    
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

            user_message = Message(
                session_id=session_id,
                role="user",
                content=request.question
            )

            assistant_message = Message(
                session_id=session_id,
                role="assistant",
                content=response.text
            )

            db.add_all([
                user_message,
                assistant_message
            ])

            db.commit()

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

@app.get("/sessions/{session_id}/messages")
def get_messages(
    session_id: str,
    db: Session = Depends(get_db)
):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    db_session = db.get(
        ConversationSession,
        session_uuid
    )

    if db_session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    messages = db.scalars(
        select(Message)
        .where(Message.session_id == session_uuid)
        .order_by(Message.id)
    ).all()

    return {
        "session_id": session_id,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at
            }
            for message in messages
        ]
    }

@app.post("/sessions/{session_id}/reset")
def reset_chat(
    session_id: str,
    db: Session = Depends(get_db)
):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    db_session = db.get(
        ConversationSession,
        session_uuid
    )

    if db_session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    db.execute(
        delete(Message)
        .where(Message.session_id == session_uuid)
    )

    db.commit()

    sessions[session_id] = create_chat()

    return {
        "message": "Conversation memory cleared."
    }

@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    db_session = db.get(
        ConversationSession,
        session_uuid
    )

    if db_session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    db.delete(db_session)
    db.commit()

    sessions.pop(session_id, None)

    return {
        "message": "Session deleted."
    }