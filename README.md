# Document Q&A Assistant

A conversational AI assistant that answers questions about uploaded documents using RAG (Retrieval Augmented Generation), LangChain, LangGraph, and FastAPI.

## What This Project Does

1. Upload any PDF or TXT document
2. Ask questions about the document in natural language
3. Get accurate AI-powered answers grounded in the document content
4. Maintains conversation history across multiple questions

## Tech Stack

- **LangChain** - Document loading and RAG pipeline
- **LangGraph** - Conversational agent workflow management
- **ChromaDB** - Local vector database for document storage
- **Groq (LLaMA 3.3)** - Free LLM for generating answers
- **FastAPI** - REST API endpoints
- **Sentence Transformers** - Text embeddings

## How It Works
User uploads document

↓

Document split into chunks

↓

Chunks stored in ChromaDB as vectors

↓

User asks a question

↓

LangGraph agent searches for relevant chunks (RAG)

↓

Groq LLM generates answer using those chunks

↓

Answer returned to user

## Project Structure
document-qa-assistant/

├── main.py              # FastAPI server and endpoints

├── rag_pipeline.py      # Document loading and vector store

├── agent.py             # LangGraph agent and Groq LLM

├── documents/           # Uploaded documents stored here

├── .env                 # API keys (not uploaded to GitHub)

└── requirements.txt     # Project dependencies

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload a document |
| POST | `/ask` | Ask a question |
| DELETE | `/reset` | Reset conversation |

## Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/Olaniyibenaya/document-qa-assistant.git
cd document-qa-assistant
```

2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your API key to `.env`
GROQ_API_KEY=your_groq_key_here
5. Run the server
```bash
uvicorn main:app --reload
```

6. Open API docs
http://127.0.0.1:8000/docs