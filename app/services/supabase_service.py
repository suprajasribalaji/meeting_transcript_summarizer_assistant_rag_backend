import os
from supabase import create_client, Client
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from typing import Optional

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL, SUPABASE_SERVICE_KEY, and SUPABASE_ANON_KEY must be set in .env")

# FIX: Create both clients once at module level — not on every request
service_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
anon_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    username: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SignupAvailabilityRequest(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    # FIX: Explicit model replaces the broken **updates pattern in auth.py
    username: Optional[str] = None


class UserProfile(BaseModel):
    id: str
    email: str
    username: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


# --------------------------------------------------------------------------- #
# Auth service
# --------------------------------------------------------------------------- #

class SupabaseAuthService:
    """Service for handling Supabase authentication."""

    @staticmethod
    async def check_signup_availability(email: Optional[str], username: Optional[str]) -> dict:
        """Check whether email and/or username are available."""
        normalized_email = (email or "").strip().lower()
        normalized_username = (username or "").strip().lower()

        email_available = True
        username_available = True

        try:
            if normalized_email:
                email_response = (
                    service_client.table("users")
                    .select("id", count="exact")
                    .ilike("email", normalized_email)
                    .limit(1)
                    .execute()
                )
                email_available = (email_response.count or 0) == 0

            if normalized_username:
                username_response = (
                    service_client.table("users")
                    .select("id", count="exact")
                    .ilike("username", normalized_username)
                    .limit(1)
                    .execute()
                )
                username_available = (username_response.count or 0) == 0

            return {
                "success": True,
                "email_available": email_available,
                "username_available": username_available,
            }
        except Exception as e:
            return {
                "success": False,
                "email_available": True,
                "username_available": True,
                "message": str(e),
            }

    @staticmethod
    async def ensure_user_profile(user_id: str, email: str, username: Optional[str] = None) -> None:
        """Create users row if missing so profile endpoints never 404 for valid accounts."""
        try:
            existing = (
                service_client.table("users")
                .select("id")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                return

            payload = {"id": user_id, "email": email}
            cleaned = (username or "").strip()
            if cleaned:
                payload["username"] = cleaned

            service_client.table("users").insert(payload).execute()
        except Exception:
            # Non-fatal: callers can still fallback to token data.
            return

    @staticmethod
    async def signup(email: str, password: str, username: str) -> dict:
        """Create a new user in Supabase Auth and insert their profile row."""
        try:
            print(f"Signup attempt: email={email}, username={username}")
            
            if len(password) < 8:
                return {"success": False, "message": "Password must be at least 8 characters"}

            if not username.strip():
                return {"success": False, "message": "Full name cannot be empty"}

            availability = await SupabaseAuthService.check_signup_availability(email, username)
            if not availability.get("success"):
                return {"success": False, "message": "Could not validate signup details. Please try again."}
            if not availability.get("email_available", True):
                return {"success": False, "message": "An account with this email already exists"}
            if not availability.get("username_available", True):
                return {"success": False, "message": "This username is already taken"}

            print("Creating Supabase user...")
            response = service_client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {"username": username.strip()}
                }
            })
            print(f"Supabase response: {response}")

            user = response.user
            if not user:
                return {"success": False, "message": "Failed to create user — no user returned"}

            print(f"Inserting user profile for id={user.id}")
            # Insert profile row (id mirrors auth.users primary key)
            # Handle RLS policy gracefully - account creation is the important part
            try:
                service_client.table("users").insert({
                    "id": user.id,
                    "email": email,
                    "username": username.strip(),
                }).execute()
                profile_message = "Account created successfully."
            except Exception as profile_error:
                print(f"Profile insertion failed (but account was created): {profile_error}")
                profile_message = "Account created successfully. You can set up your profile later."

            return {
                "success": True,
                "user_id": user.id,
                "email": user.email,
                "message": profile_message,
            }

        except Exception as e:
            print(f"Exception in signup: {e}")
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            # Surface friendly messages for common Supabase errors
            if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
                return {"success": False, "message": "An account with this email already exists"}
            return {"success": False, "message": error_msg}

    @staticmethod
    async def login(email: str, password: str) -> dict:
        """Authenticate with email + password and return tokens."""
        try:
            # FIX: Uses module-level anon_client — no new client created per request
            response = anon_client.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })

            session = response.session
            user = response.user

            if not session or not user:
                return {"success": False, "message": "Invalid email or password"}

            # Fetch profile for username (handle RLS issues gracefully)
            username = ""
            try:
                profile_response = service_client.table("users").select("username").eq("id", user.id).execute()
                if profile_response.data:
                    username = profile_response.data[0].get("username", "")
                elif user.user_metadata:
                    username = user.user_metadata.get("username", "") or ""
            except Exception as profile_error:
                print(f"Profile fetch failed (but login succeeded): {profile_error}")
                # Use username from user metadata if profile doesn't exist
                username = user.user_metadata.get("username", "") if user.user_metadata else ""

            # Best effort: ensure a profile row exists for downstream profile/settings pages.
            await SupabaseAuthService.ensure_user_profile(
                user_id=user.id,
                email=user.email or email,
                username=username or (user.user_metadata.get("username", "") if user.user_metadata else ""),
            )

            return {
                "success": True,
                "user_id": user.id,
                "email": user.email,
                "username": username,
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
            }

        except Exception as e:
            error_msg = str(e).lower()
            if "invalid login credentials" in error_msg or "invalid_credentials" in error_msg:
                return {"success": False, "message": "Invalid email or password"}
            return {"success": False, "message": "Login failed. Please try again."}

    @staticmethod
    async def verify_token(token: str) -> Optional[dict]:
        """
        Validate a Supabase JWT and return basic user info.
        FIX: Called with token from Authorization header (not query param).
        """
        try:
            response = anon_client.auth.get_user(token)
            user = response.user
            if user:
                metadata_username = ""
                if user.user_metadata:
                    metadata_username = user.user_metadata.get("username", "") or ""
                return {
                    "user_id": user.id,
                    "email": user.email,
                    "username": metadata_username,
                    "valid": True,
                }
            return None
        except Exception:
            return None

    @staticmethod
    async def logout(token: str) -> dict:
        """Revoke the Supabase session server-side."""
        try:
            # Create a temporary client scoped to this session only
            temp_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            temp_client.auth.set_session(token, "")
            temp_client.auth.sign_out()
            return {"success": True, "message": "Logged out successfully"}
        except Exception:
            return {"success": True, "message": "Logged out"}

    @staticmethod
    async def get_user_profile(user_id: str) -> Optional[dict]:
        """Fetch profile row from the users table."""
        try:
            response = service_client.table("users").select("*").eq("id", user_id).execute()
            if response.data:
                row = response.data[0]
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "username": row.get("username"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            return None
        except Exception:
            return None

    @staticmethod
    async def update_user_profile(user_id: str, updates: dict) -> dict:
        """
        Update allowed profile fields.
        FIX: Receives a clean dict from the Pydantic-validated endpoint.
        """
        try:
            # Only allow safe fields — never let raw client input overwrite id/email
            allowed_fields = {"username"}
            safe_updates = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}

            if not safe_updates:
                return {"success": False, "message": "No valid fields to update"}

            response = service_client.table("users").update(safe_updates).eq("id", user_id).execute()

            if response.data:
                return {"success": True, "message": "Profile updated successfully"}

            return {"success": False, "message": "User not found or no changes made"}

        except Exception as e:
            return {"success": False, "message": str(e)}