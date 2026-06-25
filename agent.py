import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from groq import Groq

load_dotenv()

# Connect to Groq - free and fast
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# STATE
# Shared memory that flows through every step
# of our LangGraph pipeline
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    question: str        # User's question
    context: str         # Relevant chunks from document
    answer: str          # Final generated answer
    chat_history: List   # Previous conversation messages

# ─────────────────────────────────────────────
# STEP 1 — RETRIEVE
# Search vector store for relevant document chunks
# ─────────────────────────────────────────────
def retrieve_context(state: AgentState, vector_store):
    """
    Searches the vector store for chunks
    most relevant to the user's question
    """
    question = state["question"]

    # Get top 3 most relevant chunks
    results = vector_store.similarity_search(question, k=3)

    # Combine chunks into one context block
    context = "\n\n".join([doc.page_content for doc in results])

    return {**state, "context": context}

# ─────────────────────────────────────────────
# STEP 2 — GENERATE
# Send question + context to Groq to generate answer
# ─────────────────────────────────────────────
def generate_answer(state: AgentState):
    """
    Uses retrieved context and question
    to generate a grounded answer via Groq
    """
    question = state["question"]
    context = state["context"]
    chat_history = state.get("chat_history", [])

    # Build conversation history string
    history_text = ""
    if chat_history:
        for msg in chat_history:
            history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"

    # Send to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free model on Groq
        messages=[
            {
                "role": "system",
                "content": """You are a helpful document assistant.
Use ONLY the context provided to answer questions.
If the answer is not in the context, say 
'I could not find that in the document.'"""
            },
            {
                "role": "user",
                "content": f"""Previous conversation:
{history_text}

Context from document:
{context}

Question: {question}"""
            }
        ]
    )

    answer = response.choices[0].message.content

    # Update chat history
    chat_history.append({
        "question": question,
        "answer": answer
    })

    return {**state, "answer": answer, "chat_history": chat_history}

# ─────────────────────────────────────────────
# BUILD THE GRAPH
# Connects retrieve → generate → end
# ─────────────────────────────────────────────
def build_agent(vector_store):
    """
    Builds the LangGraph agent workflow
    """
    graph = StateGraph(AgentState)

    # Add steps as nodes
    graph.add_node(
        "retrieve",
        lambda state: retrieve_context(state, vector_store)
    )
    graph.add_node("generate", generate_answer)

    # Connect nodes in order
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

def run_agent(agent, question: str, chat_history: List = []):
    """
    Runs the full agent pipeline with a question
    Returns the answer and updated chat history
    """
    initial_state = {
        "question": question,
        "context": "",
        "answer": "",
        "chat_history": chat_history
    }

    result = agent.invoke(initial_state)

    return result["answer"], result["chat_history"]