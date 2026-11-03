# AI Financial Assistant API

A small AI-powered backend application built with Python and FastAPI.

The application integrates with the Google Gemini API and provides a conversational financial education assistant with persistent PostgreSQL-backed conversation history and Retrieval-Augmented Generation (RAG).

This project was created as a hands-on learning project focused on building and integrating LLM-powered applications.

## Features

- FastAPI REST API
- Google Gemini API integration
- Custom system instructions
- Retrieval-Augmented Generation (RAG)
- Document chunking with overlap
- Semantic search using Gemini embeddings
- Top-3 relevant context retrieval
- Grounded answers based on source documents
- Per-session conversational memory with unique UUIDs
- PostgreSQL persistence for conversation sessions and messages
- Persistent chat history restored after application restart
- Session reset and deletion
- Conversation history retrieval endpoint
- SQLAlchemy ORM integration
- Alembic database migrations
- Request validation using Pydantic
- Environment-based API and model configuration
- Structured Gemini API error handling and retry logic
- Automated unit and API tests with pytest
- Docker containerization
- Docker Compose for local PostgreSQL
- GitHub Actions CI with automated tests, PostgreSQL and Docker builds
- Automatically generated Swagger/OpenAPI documentation

## Tech Stack

- Python
- FastAPI
- Google Gemini API
- Gemini Embeddings
- PostgreSQL
- SQLAlchemy
- Alembic
- Docker
- Docker Compose
- pytest
- GitHub Actions

## Project Structure

```text
fastapi-demo/
├── .github/
│   └── workflows/
│       └── ci.yml
├── alembic/
├── documents/
├── tests/
├── database.py
├── models.py
├── main.py
├── rag.py
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── pytest.ini
├── requirements.txt
├── .env.example
└── README.md
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

### 4. Configure environment variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/financial_assistant
```

Never commit your real API key to GitHub.

### 5. Start PostgreSQL

Start PostgreSQL using Docker Compose:

```bash
docker compose up -d
```

Apply database migrations:
```bash
alembic upgrade head
```

### 6. Run the application

```bash
uvicorn main:app --reload
```

The API will be available at:

http://127.0.0.1:8000

Interactive Swagger documentation:

http://127.0.0.1:8000/docs

## API Endpoints

### Conversation Sessions

- `POST /sessions` — create a new conversation session
- `POST /sessions/{session_id}/ask` — ask a question within a session
- `GET /sessions/{session_id}/messages` — retrieve persisted conversation history
- `POST /sessions/{session_id}/reset` — clear the session conversation history
- `DELETE /sessions/{session_id}` — delete the session and its messages

### POST `/sessions/{session_id}/ask`

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

Each conversation has its own unique session ID.

User and assistant messages are stored in PostgreSQL. If the application restarts, the conversation history is loaded from the database and used to reconstruct the Gemini chat context.

This allows conversations to continue across application restarts while keeping individual sessions isolated from each other.

### POST `/sessions/{session_id}/reset`

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

The application includes structured Gemini API error handling for quota limits, unavailable models and temporary service failures.

Temporary service failures are retried automatically before an error response is returned.

## Testing

The project includes automated tests for:

- document chunking;
- cosine similarity;
- session creation;
- session reset;
- session deletion;
- invalid session handling.

Run the tests with:

```bash
pytest
```

GitHub Actions automatically runs the test suite against a temporary PostgreSQL service on pull requests and changes to `main`.

## Docker

The FastAPI application can be packaged as a Docker image.

Build the image:

```bash
docker build -t financial-assistant .
```

PostgreSQL is currently started separately using Docker Compose:

```bash
docker compose up -d
```

The application itself can currently be run locally with:

```bash
uvicorn main:app --reload
```

A fully containerized FastAPI + PostgreSQL Docker Compose setup is planned.

## Current Limitations

This is currently a learning project and has several intentional limitations:

- Gemini chat objects are cached in application memory but reconstructed from PostgreSQL history after restart
- RAG embeddings are currently generated at runtime rather than stored persistently
- Only a single built-in knowledge document is currently supported
- Authentication is not implemented.
- The application currently has no separate frontend.

## Planned Improvements

Future improvements include:

- Store document embeddings persistently using PostgreSQL + pgvector
- Support multiple documents and document uploads
- Add structured logging and observability
- Deploy the application to Azure
- Add automated deployment/CD
- Build a React/TypeScript frontend
- Add tool calling / agent capabilities

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