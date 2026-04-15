from qdrant_client import QdrantClient

import os
from dotenv import load_dotenv

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

print(qdrant_client.get_collections())