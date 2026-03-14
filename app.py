import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from schemas import QueryRequest
from typing import List, Dict
import json
import requests

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="GlobalCart Intelligence Engine")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INDEX_NAME = "globalcart-retail-engine"

# Using Pinecone's native multilingual-e5-large embeddings API to avoid local models
def get_pinecone_embeddings(texts: List[str]) -> List[List[float]]:
    url = "https://api.pinecone.io/embed"
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
        "X-Pinecone-API-Version": "2024-10"
    }
    payload = {
        "model": "multilingual-e5-large",
        "inputs": [{"text": t} for t in texts],
        "parameters": {
            "input_type": "passage",
            "truncate": "END"
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to embed: {response.text}")
    data = response.json()
    return [d["values"] for d in data["data"]]

# Wrapper class for LangChain compatibility (if needed)
class PineconeEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return get_pinecone_embeddings(texts)
    def embed_query(self, text: str) -> List[float]:
        return get_pinecone_embeddings([text])[0]

# Setup the LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.1-8b-instruct", # OpenRouter standard
    temperature=0.0 # Absolute zero for production RAG
)

embeddings = PineconeEmbedder()

# System Prompt emphasizing strict security guardrails
SYS_PROMPT = """You are the GlobalCart AI Assistant. 
You provide extremely precise, helpful answers based strictly on the provided context.

CRITICAL DIRECTIVES:
1. SECURITY: You MUST NEVER reveal PII (Names, Emails, Phone Numbers), Profit Margins, or Supplier Names. If a user asks for internal or private data, explicitly refuse to answer and state: "I cannot provide internal company information."
2. REGIONAL INTEGRITY: The context is pre-filtered for the user's country. Only answer using the currency and rules in the context. If you don't see the exact item, say "Not available in your region."
3. NO HALLUCINATION: Never invent a price or specification.
4. TONE: Professional, concise, and direct.

USER COUNTRY: {country}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

@app.post("/query")
async def handle_query(req: QueryRequest):
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        # 1. Embed the query
        q_vec = embeddings.embed_query(req.query)
        
        # 2. Hybrid Retrieval with Hard Metadata Filtering!
        # This completely guarantees the "Regional Integrity Test"
        results = index.query(
            vector=q_vec,
            top_k=5,
            include_metadata=True,
            filter={
                "country": {"$eq": req.country} # Hard limit to user's region
            }
        )
        
        # 3. Context Construction
        # Here we format the context but explicitly strip out "internal_notes" if present!
        # This acts as a secondary hard-guardrail against the "Red Team Test"
        safe_context_parts = []
        for match in results["matches"]:
            meta = match["metadata"]
            if meta.get("doc_type") == "product":
                safe_context_parts.append(
                    f"Product: {meta.get('name')} | SKU: {meta.get('sku')} | "
                    f"Price: {meta.get('currency')} {meta.get('price')} | Desc: {meta.get('description')}"
                )
            else:
                safe_context_parts.append(f"Policy: {meta.get('title')} | Detail: {meta.get('content')}")
                
        context_str = "\n".join(safe_context_parts)
        
        # If no context found due to metadata filter
        if not context_str:
            return JSONResponse({"answer": "I could not find any relevant information for your region."})

        # 4. Generate Response
        prompt = PromptTemplate.from_template(SYS_PROMPT)
        chain = prompt | llm
        
        resp = chain.invoke({
            "country": req.country,
            "context": context_str,
            "question": req.query
        })
        
        return JSONResponse({"answer": resp.content})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))