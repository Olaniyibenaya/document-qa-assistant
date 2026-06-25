import streamlit as st
import requests

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="🤖",
    layout="centered"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🤖 Document Q&A Assistant")
st.caption("Upload a document and ask questions about it using AI")

# ─────────────────────────────────────────────
# SESSION STATE
# Streamlit reruns on every interaction
# Session state keeps our data between reruns
# Think of it like short term memory for the UI
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# ─────────────────────────────────────────────
# SIDEBAR — Document Upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"]
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                # Send file to our FastAPI backend
                files = {"file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )}
                
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/upload",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        st.session_state.document_uploaded = True
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success(f"✅ {uploaded_file.name} processed!")
                    else:
                        st.error("Failed to process document")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Show current document
    if st.session_state.document_uploaded:
        st.info(f"📄 Active: {st.session_state.document_name}")
        
        # Reset button
        if st.button("Reset Conversation"):
            requests.delete("http://127.0.0.1:8000/reset")
            st.session_state.chat_history = []
            st.success("Conversation reset!")

# ─────────────────────────────────────────────
# MAIN AREA — Chat Interface
# ─────────────────────────────────────────────
if not st.session_state.document_uploaded:
    # Show instructions if no document uploaded
    st.info("👈 Upload a document from the sidebar to get started")
    
    st.markdown("""
    ### How to use:
    1. **Upload** a PDF or TXT document from the sidebar
    2. Click **Process Document**
    3. **Ask questions** about your document
    4. Get **AI-powered answers** instantly
    """)

else:
    # Show chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
    
    # Chat input box at the bottom
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer from our FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/ask",
                        json={"question": question}
                    )
                    
                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        st.write(answer)
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer
                        })
                    else:
                        st.error("Failed to get answer")
                        
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")