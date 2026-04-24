# Same Groq setup as sessions.py — only the RAG retrieval is new

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from .state import AgentState

load_dotenv()

# ── Same model/key pattern as sessions.py ──
model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model=model,
    groq_api_key=api_key,
    temperature=0.3,
)

# Lazy loading - embeddings will be loaded when first needed
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)


# ── NODE 1: Retrieve relevant chunks from Qdrant ──
async def retrieve_context(state: AgentState) -> AgentState:
    collection_name = f"transcript_{state['session_id']}"

    try:
        query_vector = get_embeddings().embed_query(state["user_input"])
        results = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=8,
        )
        if results.points:
            context = "\n\n".join(
                f"[Page {p.payload.get('page_number', 'N/A')}]\n{p.payload['text']}"
                for p in results.points
            )
        else:
            context = ""
    except Exception as e:
        print(f"[retrieve_context] Qdrant error: {e}")
        context = ""

    return {**state, "retrieved_context": context}


# ── NODE 2: Generate response using RAG context + chat history ──
async def generate_response(state: AgentState) -> AgentState:
    context = state.get("retrieved_context") or ""
    history_rows = state.get("history_rows") or []

    # ── Same system prompt as sessions.py _generate_chat_reply ──
    system_prompt = f"""
        You are an AI Meeting Transcription Assistant.

        Your job is to analyze meeting transcripts that may be long, multi-page, and split into multiple chunks.

        --- CORE RULES ---

        1. CONTEXT USAGE:
        - Use ONLY the retrieved transcript context to answer.
        - The context may be incomplete or split across chunks.
        - Combine information from multiple parts if needed to form a complete answer.
        - If multiple topics are discussed, organize the answer into sections.
        - Use page numbers to maintain logical order if available.
        - Do NOT infer intentions or strategies unless explicitly stated
        - Avoid repeating phrases like "According to the meeting transcript" in every answer
        - Distinguish clearly between:
            - issues (problems/risks),
            - discussions (general topics),
            - and decisions (final outcomes)

        2. MULTI-CHUNK REASONING:
        - If the answer is spread across different parts of the transcript, merge them logically.
        - Do NOT assume missing information.
        - If partial info is available, answer with what is known clearly.

        3. WHEN INFORMATION IS MISSING:
        - If the answer is not found in the context, respond:
        "Sorry, no information available"

        4. SUMMARIZATION (IMPORTANT):
        - If the user asks for a summary:
        - Cover ALL key discussion points
        - Include decisions, actions, and important insights
        - Maintain logical flow (start → discussion → conclusion)
        - Avoid repetition
        - Keep it concise but complete

        5. FORMAT HANDLING:
        - Follow user instructions strictly (bullet points, short answer, detailed, etc.)

        6. LANGUAGE:
        - Use simple, clear English
        - Avoid unnecessary technical jargon

        7. NO HALLUCINATION:
        - Do NOT add information not present in the transcript
        - Do NOT guess or infer beyond the context

        --- CONTEXT START ---
        {context if context else "No relevant transcript context retrieved"}
        --- CONTEXT END ---
    """

    lc_messages = [SystemMessage(content=system_prompt)]

    # Rebuild conversation history (same as sessions.py)
    for row in history_rows:
        role = row.get("role")
        content = (row.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    try:
        response = await llm.ainvoke(lc_messages)
        reply = (response.content or "").strip()
    except Exception:
        reply = "The assistant hit an error generating a reply. Your messages were saved. Please try again."

    return {**state, "response": reply}