import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
import sys

load_dotenv()

# We need the user to provide API keys via .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not PINECONE_API_KEY or not OPENROUTER_API_KEY:
    print("Missing API Keys! Please ensure PINECONE_API_KEY and OPENROUTER_API_KEY are set.")
    sys.exit(1)

INDEX_NAME = "globalcart-retail-engine"

# OpenRouter acts as the proxy for the LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="anthropic/claude-3.5-sonnet", # Example standard model, configurable
    temperature=0.1
)

# Since we cannot use local models, we use Pinecone Inference for Embeddings!
pc = Pinecone(api_key=PINECONE_API_KEY)

# Let's ensure the index exists
def setup_pinecone():
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone Index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024, # Multilingual-e5-large dimension
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully!")
    else:
        print(f"Index {INDEX_NAME} already exists.")

if __name__ == "__main__":
    setup_pinecone()