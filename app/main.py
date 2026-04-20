from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.auth import router as auth_router
from app.services.sessions import router as sessions_router

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")

supabase: Client = create_client(url, key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(sessions_router)

@app.get("/")
async def root():
    return {"status": "API is running", "version": "1.0.0"}

def verify_user_token(token: str):
    try:
        user = supabase.auth.get_user(token)
        return user
    except Exception as e:
        return None

def get_user_data(user_id: str):
    response = supabase.table("users").select("*").eq("id", user_id).execute()
    return response.data[0] if response.data else None