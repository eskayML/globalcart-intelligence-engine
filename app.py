import streamlit as st
import os
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Configure Streamlit UI
st.set_page_config(page_title="GlobalCart AI Engine", page_icon="🛒", layout="centered")

st.title("🛒 GlobalCart Intelligence Engine")
st.markdown("Advanced Retail RAG System with Strict Regional & Security Guardrails.")

# Robust API Key Fetcher (Supports Streamlit Community Cloud Secrets & Local .env)
def get_api_key(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name)

PINECONE_API_KEY = get_api_key("PINECONE_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")
INDEX_NAME = "globalcart-retail-engine"

if not PINECONE_API_KEY or not OPENROUTER_API_KEY:
    st.error("Missing API Keys! Please set PINECONE_API_KEY and OPENROUTER_API_KEY in the Streamlit App Settings (Secrets) or locally.")
    st.stop()

# --- Sidebar: The Regional Integrity Test ---
st.sidebar.header("🌍 Regional Context")
country_code = st.sidebar.selectbox(
    "Select your shopping region:",
    ["GH", "ZA", "IN", "NL", "KE", "US", "UK"]
)
st.sidebar.info(f"Metadata Filtering active. You are physically locked into the **{country_code}** catalog. It is mathematically impossible to retrieve products from outside this region.")

# Initialize the modern Pinecone Client (v4.0.0+)
# This client natively supports Serverless Inference without manual HTTP requests!
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize the modern LangChain OpenRouter Client
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0.0
)

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

# --- Chat UI State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_text := st.chat_input("Ask about a product, SKU, or policy..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Render Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Executing secure retrieval via Pinecone Inference..."):
            try:
                # 1. Embed query using official Pinecone Inference SDK (v4.0.0+)
                # This guarantees zero bugs with HTTP headers or formatting
                embed_response = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[prompt_text],
                    parameters={
                        "input_type": "query",
                        "truncate": "END"
                    }
                )
                q_vec = embed_response[0].values
                
                # 2. Hard Metadata Filter on Retrieval (The Regional Integrity Test)
                index = pc.Index(INDEX_NAME)
                results = index.query(
                    vector=q_vec,
                    top_k=5,
                    include_metadata=True,
                    filter={
                        "country": {"$eq": country_code}
                    }
                )
                
                # 3. Construct Safe Context (The Red Team Security Guardrail)
                safe_context_parts = []
                for match in results["matches"]:
                    meta = match["metadata"]
                    # We EXPLICITLY strip out 'internal_notes' here before the LLM ever sees it.
                    if meta.get("doc_type") == "product":
                        safe_context_parts.append(
                            f"Product: {meta.get('name', 'Unknown')} | SKU: {meta.get('sku', 'Unknown')} | "
                            f"Price: {meta.get('currency', 'USD')} {meta.get('price', '0.00')} | Desc: {meta.get('description', '')}"
                        )
                    else:
                        safe_context_parts.append(f"Policy: {meta.get('title', 'Policy')} | Detail: {meta.get('content', '')}")
                        
                context_str = "\n".join(safe_context_parts)
                
                if not context_str:
                    final_response = f"I could not find any relevant information for your query in the **{country_code}** region."
                else:
                    # 4. Generate Answer via LangChain + OpenRouter
                    prompt_template = PromptTemplate.from_template(SYS_PROMPT)
                    chain = prompt_template | llm
                    
                    resp = chain.invoke({
                        "country": country_code,
                        "context": context_str,
                        "question": prompt_text
                    })
                    final_response = resp.content
                
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")