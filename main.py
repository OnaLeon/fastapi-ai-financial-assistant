import os
import time

from rag import (
    load_document,
    split_into_chunks,
    create_chunk_embeddings,
    find_relevant_chunks,
)
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel


# Load variables from .env
load_dotenv()

app = FastAPI()


# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

document = load_document()
chunks = split_into_chunks(document)
chunk_embeddings = create_chunk_embeddings(chunks)

# Create one chat session
# This object will remember the conversation while the server is running
chat = client.chats.create(
    model="gemini-3.8-flash",
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


@app.post("/ask")
def ask_ai(request: Question):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            relevant_chunks = find_relevant_chunks(
                request.question,
                chunk_embeddings,
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

        except Exception as e:
            print("ERROR:", e)

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "error": "AI service is temporarily unavailable. Please try again later."
                }


@app.post("/reset")
def reset_chat():
    global chat

    chat = client.chats.create(
        model="gemini-3.8-flash",
        config=types.GenerateContentConfig(
            system_instruction="""
            You are a financial education assistant.
            Explain financial concepts clearly and simply.
            Keep answers concise.
            Do not give personalized financial advice.
            """
        )
    )

    return {
        "message": "Conversation memory cleared."
    }