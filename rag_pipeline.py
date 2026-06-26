import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load environment variables
load_dotenv()

# This is the folder where documents will be stored
DOCUMENTS_FOLDER = "documents"
# This is where ChromaDB saves our vector database locally
CHROMA_DB_PATH = "chroma_db"

def load_document(file_path: str):
    """
    Loads a document from disk.
    Supports PDF and TXT files.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Only PDF and TXT files are supported")
    
    # Returns a list of pages/sections from the document
    return loader.load()

def split_documents(documents):
    """
    Splits documents into smaller chunks.
    Why? Because AI models can only read small pieces at a time.
    Think of it like cutting a book into individual paragraphs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Each chunk is 500 characters
        chunk_overlap=50     # Chunks overlap by 50 chars so context isn't lost
    )
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    """
    Converts text chunks into numbers (vectors) and stores them.
    Why vectors? Because computers can search numbers much faster than text.
    Think of it like giving every paragraph a unique GPS coordinate.
    """
    # SentenceTransformer converts text into vectors (numbers)
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Chroma stores those vectors locally on your machine
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    return vector_store

def load_existing_vector_store():
    """
    Loads an already existing vector store from disk.
    So we don't rebuild it every time.
    """
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

def build_rag_pipeline(file_path: str):
    """
    MAIN FUNCTION — runs the full RAG setup:
    1. Load document
    2. Split into chunks
    3. Build vector store
    Returns the vector store ready for searching
    """
    print(f"Loading document: {file_path}")
    documents = load_document(file_path)
    
    print(f"Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print(f"Building vector store...")
    vector_store = build_vector_store(chunks)
    print(f"Vector store ready!")
    
    return vector_store