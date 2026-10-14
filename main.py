import os
import time

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


# Create one chat session
# This object will remember the conversation while the server is running
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
            response = chat.send_message(
                message=request.question
            )

            return {
                "answer": response.text
            }

        except Exception:
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