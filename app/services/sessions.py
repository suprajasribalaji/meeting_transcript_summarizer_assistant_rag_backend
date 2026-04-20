import os
from io import BytesIO
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from pypdf import PdfReader
from app.services.supabase_service import service_client, SupabaseAuthService
from app.services.supabase_storage import (
    upload_meeting_pdf_bytes,
    delete_meeting_pdf_for_user,
    download_object_bytes_by_file_url,
)
from langchain_groq import ChatGroq
from app.agents.graph import transcript_agent
from app.services.vector_store import index_transcript, delete_transcript_index
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
api_key = os.getenv("GROQ_API_KEY")

RESERVED_SESSION_IDS = frozenset({"history", "upload-pdf"})

router = APIRouter(prefix="/sessions", tags=["sessions"])
bearer_scheme = HTTPBearer()

MAX_PDF_BYTES = 10 * 1024 * 1024


async def _summarize_transcript(text: str) -> str:
    """Generate an initial summary using Groq when configured."""
    if not api_key or not text.strip():
        excerpt = text.strip()[:1200]
        if len(text.strip()) > 1200:
            excerpt += "\n\n[…]"
        return (
            "Here is the extracted text from your PDF (configure GROQ_API_KEY for an AI summary):\n\n"
            + excerpt
        )

    try:
        llm = ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=0.3,
        )
        system_prompt = """
You summarize meeting transcripts. Reply with:
- A short opening line confirming the transcript was received.
- EXACTLY 5 concise bullet points of key themes, decisions, or action items inferred from the text.
- End with one line inviting follow-up questions.
Keep a professional tone. Base answers ONLY on the transcript; if it is unclear, say so briefly.
"""
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text[:50_000]),
            ]
        )
        return (response.content or "").strip()
    except Exception:
        excerpt = text.strip()[:1200]
        if len(text.strip()) > 1200:
            excerpt += "\n\n[…]"
        return (
            "Summary could not be generated automatically. Here is an excerpt from your PDF:\n\n"
            + excerpt
        )


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "".join(parts).strip()


def _download_transcript_from_file_url(file_url: str) -> str:
    """
    Use Supabase Storage API (service role) for hosted file URLs so private buckets work.
    HTTP GET on /object/public/... returns 400 when the object is not publicly readable.
    """
    if "/storage/v1/object/" in file_url:
        raw = download_object_bytes_by_file_url(file_url)
        return _extract_text_from_pdf_bytes(raw)

    r = requests.get(file_url, timeout=120)
    r.raise_for_status()
    return _extract_text_from_pdf_bytes(r.content)


def _get_transcript_for_session(session: dict) -> str:
    raw = session.get("transcript")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    url = session.get("file_url")
    if not url:
        return ""
    return _download_transcript_from_file_url(url)


async def _generate_chat_reply(transcript: str, history_rows: list[dict]) -> str:
    """Multi-turn reply grounded in the meeting transcript."""
    truncated = transcript[:50_000] if transcript else ""

    if not api_key:
        return (
            "Configure GROQ_API_KEY in the server environment for AI answers. "
            "Your messages were still saved."
        )

    try:
        llm = ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=0.3,
        )
        system_prompt = f"""
You are an AI Meeting Transcription Assistant.

Rules:
- Answer based STRICTLY on the transcript when the user asks about the meeting.
- If the user specifies format, length, or bullet points, follow their instructions.
- If you cannot find the answer in the transcript, say you don't know.
- Stay professional and concise. Avoid filler.

Transcript:
{truncated}
"""
        lc_messages: list = [SystemMessage(content=system_prompt)]
        for row in history_rows:
            role = row.get("role")
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        response = await llm.ainvoke(lc_messages)
        return (response.content or "").strip()
    except Exception:
        return (
            "The assistant hit an error generating a reply. Your messages were saved. "
            "Please try again."
        )


class ChatMessageBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=16_000)


# --------------------------------------------------------------------------- #
# Auth dependency (same pattern as auth.py)
# --------------------------------------------------------------------------- #

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials
    user = await SupabaseAuthService.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@router.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    """
    Fetch all sessions for the logged-in user,
    ordered by most recently updated, with message count.
    """
    try:
        user_id = current_user["user_id"]

        # Fetch sessions for this user
        sessions_response = service_client.table("sessions") \
            .select("id, title, file_name, file_url, created_at, updated_at") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()

        sessions = sessions_response.data or []

        if not sessions:
            return JSONResponse(status_code=200, content={"success": True, "sessions": []})

        # Fetch message counts for each session
        session_ids = [s["id"] for s in sessions]

        messages_response = service_client.table("messages") \
            .select("session_id") \
            .in_("session_id", session_ids) \
            .execute()

        # Count messages per session
        message_counts: dict[str, int] = {}
        for msg in (messages_response.data or []):
            sid = msg["session_id"]
            message_counts[sid] = message_counts.get(sid, 0) + 1

        # Merge counts into sessions
        result = []
        for session in sessions:
            result.append({
                "id": session["id"],
                "title": session["title"],
                "file_name": session.get("file_name"),
                "file_url": session.get("file_url"),
                "message_count": message_counts.get(session["id"], 0),
                "updated_at": session["updated_at"],
                "created_at": session["created_at"],
            })

        return JSONResponse(status_code=200, content={"success": True, "sessions": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Accept a meeting transcript PDF, extract text, create a session,
    persist an upload message + summary, and return identifiers for the client.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        suffix = Path(file.filename).suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        content_type = (file.content_type or "").lower()
        if content_type and "pdf" not in content_type:
            raise HTTPException(status_code=400, detail="File must be a PDF")

        raw = await file.read()
        if len(raw) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

        reader = PdfReader(BytesIO(raw))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        transcript = "".join(text_parts).strip()

        if not transcript:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from this PDF. Try another file or export.",
            )

        user_id = current_user["user_id"]
        safe_name = Path(file.filename).name
        title = Path(file.filename).stem or "Meeting transcript"

        try:
            file_url, _storage_path = upload_meeting_pdf_bytes(user_id, safe_name, raw)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not upload file to Supabase Storage: {exc}",
            ) from exc

        summary = await _summarize_transcript(transcript)

        # Index transcript into Qdrant for RAG (session_id set after session row created)
        # We'll index after we have the session_id below

        base_session = {
            "user_id": user_id,
            "title": title[:500],
            "file_name": safe_name[:500],
            "file_url": file_url,
        }
        try:
            session_row = (
                service_client.table("sessions")
                .insert({**base_session, "transcript": transcript[:800_000]})
                .execute()
            )
        except Exception:
            session_row = service_client.table("sessions").insert(base_session).execute()

        if not session_row.data:
            raise HTTPException(status_code=500, detail="Failed to create session")

        session_id = session_row.data[0]["id"]

        # Index into Qdrant for RAG
        try:
            index_transcript(session_id, transcript)
        except Exception as e:
            print(f"Qdrant indexing failed (non-fatal): {e}")

        service_client.table("messages").insert(
            [
                {
                    "session_id": session_id,
                    "role": "user",
                    "content": f"📄 Uploaded: {safe_name}",
                },
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": summary,
                },
            ]
        ).execute()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "session_id": session_id,
                "file_name": safe_name,
                "file_url": file_url,
                "summary": summary,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/chat")
async def post_chat_message(
    session_id: str,
    body: ChatMessageBody,
    current_user: dict = Depends(get_current_user),
):
    """
    Append a user message, run the assistant on the transcript + history,
    append the assistant reply, and return both rows.
    """
    if session_id in RESERVED_SESSION_IDS:
        raise HTTPException(status_code=404, detail="Session not found")

    user_id = current_user["user_id"]
    text = body.message.strip()

    try:
        session_resp = (
            service_client.table("sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not session_resp.data:
            raise HTTPException(status_code=404, detail="Session not found")

        session = session_resp.data[0]

        try:
            transcript = _get_transcript_for_session(session)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not load transcript: {exc}",
            ) from exc

        if not transcript:
            raise HTTPException(
                status_code=400,
                detail="No transcript available for this session. Upload a PDF first.",
            )

        user_insert = (
            service_client.table("messages")
            .insert(
                {
                    "session_id": session_id,
                    "role": "user",
                    "content": text,
                }
            )
            .execute()
        )
        if not user_insert.data:
            raise HTTPException(status_code=500, detail="Failed to save user message")
        user_row = user_insert.data[0]

        history_resp = (
            service_client.table("messages")
            .select("id, role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        history_rows = history_resp.data or []

        # LangGraph RAG agent — replaces _generate_chat_reply
        agent_result = await transcript_agent.ainvoke({
            "session_id": session_id,
            "user_input": text,
            "history_rows": history_rows,
            "retrieved_context": None,
            "response": None,
        })
        assistant_text = agent_result["response"]

        asst_insert = (
            service_client.table("messages")
            .insert(
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": assistant_text,
                }
            )
            .execute()
        )
        if not asst_insert.data:
            raise HTTPException(status_code=500, detail="Failed to save assistant message")
        assistant_row = asst_insert.data[0]

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user_message": user_row,
                "assistant_message": assistant_row,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a session: remove the PDF from Supabase Storage (when applicable),
    then delete all messages and the session row. Ownership required.
    """
    if session_id in RESERVED_SESSION_IDS:
        raise HTTPException(status_code=404, detail="Session not found")

    user_id = current_user["user_id"]

    try:
        session_resp = (
            service_client.table("sessions")
            .select("id, file_url")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not session_resp.data:
            raise HTTPException(status_code=404, detail="Session not found")

        file_url = session_resp.data[0].get("file_url")

        try:
            delete_meeting_pdf_for_user(user_id, file_url)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not delete file from Supabase Storage: {exc}",
            ) from exc

        service_client.table("messages").delete().eq("session_id", session_id).execute()

        # Clean up Qdrant collection for this session
        try:
            delete_transcript_index(session_id)
        except Exception as e:
            print(f"Qdrant cleanup failed (non-fatal): {e}")

        service_client.table("sessions").delete().eq("id", session_id).eq(
            "user_id", user_id
        ).execute()

        return JSONResponse(status_code=200, content={"success": True})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Session metadata only (no messages). Use GET /{session_id}/messages for full thread.
    """
    if session_id in RESERVED_SESSION_IDS:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        user_id = current_user["user_id"]
        session_response = (
            service_client.table("sessions")
            .select("id, title, file_name, file_url, created_at, updated_at")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not session_response.data:
            raise HTTPException(status_code=404, detail="Session not found")

        return JSONResponse(
            status_code=200,
            content={"success": True, "session": session_response.data[0]},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Fetch all messages for a specific session.
    Ownership check — users can only access their own sessions.
    """
    try:
        user_id = current_user["user_id"]

        # Verify session belongs to this user
        session_response = service_client.table("sessions") \
            .select("id, title, file_name, file_url") \
            .eq("id", session_id) \
            .eq("user_id", user_id) \
            .execute()

        if not session_response.data:
            raise HTTPException(status_code=404, detail="Session not found")

        session = session_response.data[0]

        # Fetch messages ordered chronologically
        messages_response = service_client.table("messages") \
            .select("id, role, content, created_at") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()

        return JSONResponse(status_code=200, content={
            "success": True,
            "session": session,
            "messages": messages_response.data or [],
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))