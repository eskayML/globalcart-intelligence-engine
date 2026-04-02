import streamlit as st
import os
import time
import io
import uuid
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
from swarm import Swarm, Agent

try:
    import speech_recognition as sr
    from gtts import gTTS
except ImportError:
    st.error("Please run: uv pip install SpeechRecognition gTTS pydub pandas")
    st.stop()

# --- ⚙️ Application & Data Setup ---

st.set_page_config(page_title="GlobalCart Intelligence Engine", page_icon="🛒", layout="centered", initial_sidebar_state="collapsed")
st.title("🛒 GlobalCart Intelligence Engine")

# --- 📝 System Description (Professional Summary) ---
st.markdown("""
**GlobalCart** is a specialized retail intelligence system designed to streamline your shopping experience through a coordinated multi-agent architecture:

*   **Sales Orchestrator:** Seamlessly manages your session, remembers your context, and routes your needs to the right specialist.
*   **Inventory Analyst:** Executes real-time data science on our 3,000+ item global inventory to find the exact prices, counts, and minimums you need.
*   **Product Specialist:** A semantic RAG system that understands the nuances of product specs, store policies, and regional availability.

*Experience a smarter way to shop—just ask or speak your mind.*
""")

# --- SESSION & THREAD MANAGEMENT ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "swarm_history" not in st.session_state:
    st.session_state.swarm_history = []
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

st.sidebar.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")

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
    st.error("Missing API Keys! Please set PINECONE_API_KEY and OPENROUTER_API_KEY.")
    st.stop()

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
swarm_client = Swarm(client=client)

AGENT_MODEL = "google/gemini-2.0-flash-001"

# --- 📊 Load Cleaned Local Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("inventory.csv")
        bad_cols = ["Internal_Notes", "Profit_Margin", "Supplier"]
        for col in bad_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return None

df_global = load_data()

# --- 🤖 Swarm Agent Functions ---

def run_pandas_query(query_expression: str):
    try:
        result = eval(query_expression, {"__builtins__": {}}, {"df": df_global})
        return str(result)
    except Exception as e:
        return f"Query Error: {str(e)}"

def retrieve_retail_knowledge(query: str):
    try:
        embed_response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )
        q_vec = embed_response[0].values
        index = pc.Index(INDEX_NAME)
        results = index.query(vector=q_vec, top_k=5, include_metadata=True)
        
        parts = []
        for m in results["matches"]:
            meta = m["metadata"]
            if meta.get("doc_type") == "product":
                parts.append(f"Product: {meta.get('name')} | Price: {meta.get('price')} | Specs: {meta.get('specs')}")
            else:
                parts.append(f"Policy: {meta.get('title')} | Detail: {meta.get('content')}")
        return "\n".join(parts) if parts else "No specific matches found in knowledge base."
    except Exception as e:
        return f"RAG Error: {str(e)}"

# --- 🤖 Swarm Agent Definitions ---

data_analyst = Agent(
    name="Data Analyst",
    instructions="""You are the GlobalCart Data Analyst. 
    Translate user questions into read-only pandas expressions. 
    The dataframe is named 'df'. Columns: Product_ID, Country, Category, Item_Name, Price_Local, Currency, Technical_Specs.
    Return ONLY the result of the function 'run_pandas_query'. 
    If the user hasn't specified a country, ask them first unless the query is global.""",
    functions=[run_pandas_query],
    model=AGENT_MODEL
)

rag_specialist = Agent(
    name="RAG Specialist",
    instructions="""You are the GlobalCart Semantic Specialist. 
    Use 'retrieve_retail_knowledge' to answer questions about product features, specs, and store policies.
    If the user asks for prices across multiple items or math, handoff to the Data Analyst.""",
    functions=[retrieve_retail_knowledge],
    model=AGENT_MODEL
)

planner_agent = Agent(
    name="GlobalCart Planner",
    instructions="""You are the lead sales orchestrator for GlobalCart.
    Your job is to coordinate between the Data Analyst and the RAG Specialist.
    1. Greeting: Handle politely yourself as a professional sales assistant.
    2. Numerical/Math/Comparisons: Transfer to Data Analyst.
    3. General Knowledge/Specs/Policy: Transfer to RAG Specialist.
    4. MANDATORY: You MUST maintain conversation history. If the user already told you their country, don't ask again.
    5. IDENTITY: You are a professional retail intelligence system. No robotic slop. Never mention internal names or your developer.""",
    model=AGENT_MODEL
)

def transfer_to_analyst(): return data_analyst
def transfer_to_rag(): return rag_specialist

planner_agent.functions = [transfer_to_analyst, transfer_to_rag]

# --- 💬 Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# --- 🎤 Input Handling ---
prompt_text = None
is_voice = False

if hasattr(st, "audio_input"):
    audio_value = st.audio_input("🎙️ Speak your message", key=f"voice_input_{st.session_state.audio_key}")
else:
    audio_value = st.file_uploader("Upload Audio", type=["wav", "mp3"], key=f"voice_input_{st.session_state.audio_key}")

text_input = st.chat_input("Ask about products, prices, or policies...")

if audio_value:
    with st.spinner("Transcribing..."):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
                prompt_text = r.recognize_google(audio_data)
                is_voice = True
        except: st.error("Transcription failed.")
elif text_input:
    prompt_text = text_input

# --- 🚀 Swarm Execution ---
if prompt_text:
    display_user_msg = f"🗣️ {prompt_text}" if is_voice else prompt_text
    st.session_state.messages.append({"role": "user", "content": display_user_msg})
    st.session_state.swarm_history.append({"role": "user", "content": prompt_text})
    
    with st.chat_message("user"):
        st.markdown(display_user_msg)

    with st.chat_message("assistant"):
        with st.spinner("GlobalCart Brain Thinking..."):
            response = swarm_client.run(
                agent=planner_agent,
                messages=st.session_state.swarm_history
            )
            
            full_response = response.messages[-1]["content"]
            st.session_state.swarm_history = response.messages
            st.markdown(full_response)
            
            audio_bytes = None
            if is_voice:
                with st.spinner("Generating voice response..."):
                    tts = gTTS(text=full_response, lang='en')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    audio_bytes = fp.getvalue()
                    st.audio(audio_bytes, format="audio/mp3")

            msg_entry = {"role": "assistant", "content": full_response}
            if audio_bytes:
                msg_entry["audio"] = audio_bytes
            st.session_state.messages.append(msg_entry)

            if is_voice:
                st.session_state.audio_key += 1
                st.rerun()
