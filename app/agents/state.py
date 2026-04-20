from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    user_input: str
    history_rows: List[dict]       # fetched from Supabase messages table
    retrieved_context: Optional[str]
    response: Optional[str]