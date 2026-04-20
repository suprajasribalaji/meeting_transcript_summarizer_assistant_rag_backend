import uuid
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

SIZE = 384

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


def _ensure_collection(collection_name: str):
    existing = {c.name for c in qdrant.get_collections().collections}
    if collection_name not in existing:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=SIZE, distance=Distance.COSINE),
        )


def index_transcript(session_id: str, transcript_text: str) -> int:
    """
    Splits the transcript text into chunks and indexes into Qdrant.
    Called once from sessions.py upload_pdf after text extraction.
    Returns number of chunks indexed.
    """
    collection_name = f"transcript_{session_id}"
    _ensure_collection(collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    chunks = splitter.split_text(transcript_text)
    chunks = [c for c in chunks if c.strip()]

    if not chunks:
        return 0

    vectors = get_embeddings().embed_documents(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "session_id": session_id},
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    qdrant.upsert(collection_name=collection_name, points=points)
    return len(chunks)


def delete_transcript_index(session_id: str):
    """
    Deletes the Qdrant collection for a session.
    Call from sessions.py delete_session for cleanup.
    """
    collection_name = f"transcript_{session_id}"
    existing = {c.name for c in qdrant.get_collections().collections}
    if collection_name in existing:
        qdrant.delete_collection(collection_name)