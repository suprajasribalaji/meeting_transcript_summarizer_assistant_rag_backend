from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.services.supabase_service import (
    SupabaseAuthService,
    SignupRequest,
    LoginRequest,
    UpdateProfileRequest,
)

router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer()


# --------------------------------------------------------------------------- #
# Auth dependency — protects any route that needs a valid logged-in user
# --------------------------------------------------------------------------- #

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    FIX: Reusable dependency that validates the Bearer token on every
    protected request. Attach with `current_user: dict = Depends(get_current_user)`.
    """
    token = credentials.credentials
    user = await SupabaseAuthService.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user  # {"user_id": ..., "email": ..., "valid": True}


# --------------------------------------------------------------------------- #
# Public routes (no auth required)
# --------------------------------------------------------------------------- #

@router.get("/")
async def home():
    return {"message": "Auth API is working"}


@router.post("/signup")
async def signup(request: SignupRequest):
    """Register a new user."""
    try:
        result = await SupabaseAuthService.signup(
            email=request.email,
            password=request.password,
            username=request.username,
        )
        if result["success"]:
            return JSONResponse(status_code=201, content=result)
        raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        print(f"Signup error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(request: LoginRequest):
    """Authenticate and return access + refresh tokens."""
    result = await SupabaseAuthService.login(
        email=request.email,
        password=request.password,
    )
    if result["success"]:
        return JSONResponse(status_code=200, content=result)
    raise HTTPException(status_code=401, detail=result["message"])


# --------------------------------------------------------------------------- #
# Protected routes (valid Bearer token required)
# --------------------------------------------------------------------------- #

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user),
                 credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Invalidate the Supabase session server-side.
    FIX: New endpoint so the client can cleanly revoke the token, not just
    delete it from localStorage.
    """
    result = await SupabaseAuthService.logout(credentials.credentials)
    return JSONResponse(status_code=200, content=result)


@router.post("/verify-token")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """
    FIX: Token now comes from the Authorization header (Bearer <token>),
    not a query param — query params appear in server logs and browser history.
    """
    return JSONResponse(status_code=200, content=current_user)


@router.get("/profile/{user_id}")
async def get_profile(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get a user profile.
    FIX: Protected — requires valid token. Users can only fetch their own profile.
    """
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="You can only access your own profile")

    profile = await SupabaseAuthService.get_user_profile(user_id)
    if profile:
        return JSONResponse(status_code=200, content=profile)
    raise HTTPException(status_code=404, detail="User not found")


@router.put("/profile/{user_id}")
async def update_profile(
    user_id: str,
    body: UpdateProfileRequest,           # FIX: explicit Pydantic model — **updates was completely broken
    current_user: dict = Depends(get_current_user),
):
    """
    Update a user profile.
    FIX: Protected + ownership check + proper request body model.
    """
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="You can only update your own profile")

    result = await SupabaseAuthService.update_user_profile(user_id, body.model_dump(exclude_none=True))
    if result["success"]:
        return JSONResponse(status_code=200, content=result)
    raise HTTPException(status_code=400, detail=result["message"])