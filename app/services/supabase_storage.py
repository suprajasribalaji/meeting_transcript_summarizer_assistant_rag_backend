"""Upload and delete meeting PDFs in Supabase Storage (service role)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.services.supabase_service import service_client


def _bucket_name() -> str:
    name = os.getenv("SUPABASE_STORAGE_BUCKET", "").strip()
    if not name:
        raise RuntimeError(
            "SUPABASE_STORAGE_BUCKET is not set — create a bucket in the Supabase dashboard "
            "and add its name to .env"
        )
    return name


def _safe_filename(name: str) -> str:
    base = Path(name).name
    out = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    return out[:200] or "document.pdf"


def download_object_bytes_by_file_url(file_url: str) -> bytes:
    """
    Download file bytes using the Storage API (service role).
    Use this instead of HTTP GET on public URLs — private buckets return 400 on /object/public/... links.
    """
    bucket = _bucket_name()
    path = object_path_from_supabase_url(file_url, bucket)
    if not path:
        raise RuntimeError("Could not parse storage object path from file URL")

    data = service_client.storage.from_(bucket).download(path)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if hasattr(data, "read"):
        return data.read()
    return bytes(data)


def object_path_from_supabase_url(file_url: str, bucket: str) -> str | None:
    """Extract storage object path from a Supabase Storage public or signed URL."""
    if not file_url:
        return None
    try:
        parsed = urlparse(file_url.strip())
        path = unquote(parsed.path)
        for marker in (f"/object/public/{bucket}/", f"/object/sign/{bucket}/"):
            if marker in path:
                return path.split(marker, 1)[1]
    except Exception:
        pass
    return None


def upload_meeting_pdf_bytes(user_id: str, original_filename: str, data: bytes) -> tuple[str, str]:
    """
    Upload PDF bytes to Supabase Storage.
    Returns (file_url, storage_path). URL is public or long-lived signed, depending on settings.
    """
    bucket = _bucket_name()
    safe = _safe_filename(original_filename)
    object_id = str(uuid.uuid4())
    storage_path = f"meeting_transcripts/{user_id}/{object_id}_{safe}"

    storage = service_client.storage.from_(bucket)
    storage.upload(
        storage_path,
        data,
        file_options={"content-type": "application/pdf"},
    )

    use_public = os.getenv("SUPABASE_STORAGE_PUBLIC", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_public:
        pub = storage.get_public_url(storage_path)
        if isinstance(pub, str) and pub.startswith("http"):
            file_url = pub
        elif isinstance(pub, dict):
            file_url = pub.get("publicUrl") or pub.get("publicURL") or ""
        else:
            file_url = str(pub) if pub else ""
        if file_url:
            return file_url, storage_path

    # Private bucket or fallback: long-lived signed URL (seconds)
    signed = storage.create_signed_url(storage_path, 60 * 60 * 24 * 365)
    if isinstance(signed, dict):
        file_url = signed.get("signedURL") or signed.get("signedUrl") or ""
    else:
        file_url = str(signed) if signed else ""

    if not file_url:
        raise RuntimeError("Could not build a URL for the uploaded file (check Storage policies).")

    return file_url, storage_path


def delete_meeting_pdf_for_user(user_id: str, file_url: str | None) -> None:
    """Remove the object from Supabase Storage if the URL belongs to this user's prefix."""
    if not file_url:
        return

    bucket = _bucket_name()
    object_path = object_path_from_supabase_url(file_url, bucket)
    if not object_path:
        return

    prefix = f"meeting_transcripts/{user_id}/"
    if not object_path.startswith(prefix):
        raise PermissionError("Stored file path does not belong to this user.")

    try:
        service_client.storage.from_(bucket).remove([object_path])
    except Exception as exc:
        name = type(exc).__name__
        msg = str(exc).lower()
        if "not_found" in name or "not found" in msg or "404" in msg or "no such" in msg:
            return
        raise
