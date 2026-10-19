# AI Financial Assistant API

A small AI-powered backend application built with Python and FastAPI.

The application integrates with the Google Gemini API and provides a conversational financial education assistant with temporary conversation memory and Retrieval-Augmented Generation (RAG).

This project was created as a hands-on learning project focused on building and integrating LLM-powered applications.

## Features

- FastAPI REST API
- Google Gemini API integration
- Custom system instructions
- Conversational memory during the current session
- Conversation reset endpoint
- Retrieval-Augmented Generation (RAG)
- Document chunking with overlap
- Semantic search using embeddings
- Top-3 relevant context retrieval
- Grounded answers based on source documents
- Automatic retry handling for temporary AI service failures
- Request validation using Pydantic
- Environment-based API key configuration
- Automatically generated Swagger/OpenAPI documentation
- Automated unit tests with pytest
- Docker containerization
- Environment-based Gemini model configuration
- Structured Gemini API error handling
- GitHub Actions CI for automated tests and Docker builds

## Tech Stack

- Python
- FastAPI
- Google Gemini API
- Gemini Embeddings
- Pydantic
- Uvicorn
- python-dotenv
- Docker

## Project Structure

```text
fastapi-ai-financial-assistant/
├── documents/
│   └── financial_guide.txt
├── main.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
└── .venv/
```

The `.env` file and `.venv` directory are excluded from Git and should not be committed.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd fastapi-ai-financial-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit your real API key to GitHub.

### 5. Run the application

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### POST `/ask`

Sends a question to the AI assistant.

Example request:

```json
{
  "question": "What is compound interest?"
}
```

Example response:

```json
{
  "answer": "Compound interest means earning interest on both the original amount and previously accumulated interest.",
  "source": "financial_guide.txt",
  "retrieved_chunks": 3
}
```

The assistant uses a system instruction that tells it to:

- explain financial concepts clearly;
- keep responses concise;
- act as a financial education assistant;
- avoid personalized financial advice;
- answer using the retrieved document context.

### Conversation Memory

The assistant remembers previous messages during the current server session.

For example:

First request:

```json
{
  "question": "My name is Ona and I have 10000 euros."
}
```

Second request:

```json
{
  "question": "How much money did I say I have?"
}
```

The assistant can answer using information from the previous message.

### POST `/reset`

Clears the current conversation memory and starts a new chat session.

Example response:

```json
{
  "message": "Conversation memory cleared."
}
```

## RAG Architecture

The application uses Retrieval-Augmented Generation (RAG) to answer questions using information from a local financial guide.

The document is:

1. Loaded from a text file.
2. Split into overlapping chunks.
3. Converted into vector embeddings using Gemini Embeddings.
4. Compared with the user's question using cosine similarity.
5. The top 3 most relevant chunks are added to the LLM context.
6. Gemini generates an answer grounded in the retrieved document content.

This allows the assistant to answer based on the provided source material instead of relying only on the model's general knowledge.

## Error Handling

The application includes basic retry logic for temporary AI service failures.

If the Gemini API is temporarily unavailable, the application retries the request before returning an error response.

## Testing

The project includes automated unit tests for core RAG functionality, including:

- document chunking;
- cosine similarity;
- semantic similarity calculations.

Run the tests with:

```bash
pytest

## Docker

The application can be built and run inside a Docker container.

Build the image:

```bash
docker build -t financial-assistant .
```

Run the container with the Gemini API key from the local `.env` file:

```bash
docker run --env-file .env -p 8000:8000 financial-assistant
```

The Swagger API documentation is then available at:

```text
http://127.0.0.1:8000/docs
```

## Current Limitations

This is currently a learning project and has several intentional limitations:

- Conversation history is stored only in application memory.
- Conversation history is lost when the server restarts.
- All users currently share the same conversation session.
- Document embeddings are generated at application startup.
- No persistent database is used yet.
- Authentication is not implemented.
- The application currently has no separate frontend.

## Planned Improvements

Future improvements include:

- Separate chat sessions for each user
- Persistent conversation storage
- Automated deployment pipeline
- Persistent vector storage
- Support for multiple uploaded documents
- Logging and observability
- Azure deployment

## Learning Goals

The project is being used to gain practical experience with:

- Python backend development
- REST API development
- FastAPI
- LLM API integration
- Prompt and system instruction design
- Conversational AI
- Retrieval-Augmented Generation (RAG)
- Embeddings and semantic search
- Error handling and retries
- Production-oriented AI application development
- Automated testing with pytest