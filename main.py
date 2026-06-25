import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from rag_pipeline import build_rag_pipeline, load_existing_vector_store
from agent import build_agent, run_agent

load_dotenv()

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Document Q&A Assistant",
    description="Upload a document and ask questions about it using RAG and LangGraph",
    version="1.0.0"
)

# Allows frontend apps to talk to our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables to store our agent and chat history
# In production you'd use a database — for now this works
current_agent = None
current_chat_history = []

# ─────────────────────────────────────────────
# REQUEST MODELS
# These define what data each endpoint expects
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    chat_history: List

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def home():
    """Health check — confirms API is running"""
    return {
        "status": "running",
        "message": "Document Q&A Assistant is ready"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint 1: Upload a document
    - Accepts PDF or TXT files
    - Builds the RAG pipeline
    - Returns confirmation
    """
    global current_agent, current_chat_history
    
    # Validate file type
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )
    
    # Save uploaded file to documents folder
    file_path = f"documents/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Build RAG pipeline from the uploaded document
    print(f"Building RAG pipeline for {file.filename}...")
    vector_store = build_rag_pipeline(file_path)
    
    # Build LangGraph agent with the vector store
    current_agent = build_agent(vector_store)
    
    # Reset chat history for new document
    current_chat_history = []
    
    return {
        "message": f"Document '{file.filename}' uploaded successfully",
        "status": "ready"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint 2: Ask a question about the uploaded document
    - Takes a question
    - Runs through LangGraph agent
    - Returns answer + chat history
    """
    global current_agent, current_chat_history
    
    # Make sure a document has been uploaded first
    if current_agent is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a document first."
        )
    
    # Run the agent
    answer, current_chat_history = run_agent(
        current_agent,
        request.question,
        current_chat_history
    )
    
    return {
        "answer": answer,
        "chat_history": current_chat_history
    }

@app.delete("/reset")
def reset_conversation():
    """
    Endpoint 3: Reset the conversation history
    Keeps the document but clears chat history
    """
    global current_chat_history
    current_chat_history = []
    return {"message": "Conversation reset successfully"}