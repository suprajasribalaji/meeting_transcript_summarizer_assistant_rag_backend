from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.services.supabase_service import service_client, SupabaseAuthService

router = APIRouter(prefix="/sessions", tags=["sessions"])
bearer_scheme = HTTPBearer()


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