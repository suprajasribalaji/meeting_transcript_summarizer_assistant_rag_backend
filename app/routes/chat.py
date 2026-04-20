import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.agents.graph import transcript_agent
from app.services.vector_store import index_transcript, collection_has_docs
from app.routes.auth import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    file_uploaded: bool


# Replaces: @cl.on_message PDF upload block
@router.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunk_count = index_transcript(tmp_path)
    finally:
        os.unlink(tmp_path)

    if chunk_count == 0:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    return {"success": True, "message": "✅ PDF indexed successfully! Ask me anything.", "chunks": chunk_count}


# Replaces: @cl.on_message normal chat flow
@router.post("/message", response_model=ChatResponse)
async def chat_message(
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    file_uploaded = collection_has_docs()

    result = await transcript_agent.ainvoke({
        "messages": [HumanMessage(content=body.message)],
        "user_input": body.message.strip() or "Provide a concise summary in 5 bullet points.",
        "session_id": str(current_user.get("id", "")),
        "file_uploaded": file_uploaded,
        "retrieved_context": None,
        "response": None,
    })

    return ChatResponse(response=result["response"], file_uploaded=file_uploaded)