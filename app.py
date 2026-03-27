import streamlit as st
import os
import time
import io
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

try:
    import speech_recognition as sr
    from gtts import gTTS
except ImportError:
    st.error("Please run: uv pip install SpeechRecognition gTTS pydub pandas")
    st.stop()

# --- 🛡️ Guardrail Definitions ---

def input_guardrail(prompt_text: str) -> bool:
    """
    STUB: Input Guardrail Evaluator.
    Intercepts prompt injections or prohibited queries before they reach any agent.
    """
    restricted_keywords = ["ignore previous instructions", "system prompt", "bypass", "drop table", "os.system"]
    if any(keyword in prompt_text.lower() for keyword in restricted_keywords):
        return False
    return True

def output_guardrail(response_text: str) -> str:
    """
    STUB: Output Guardrail Filter.
    Scans the generated response for PII, supplier leaks, or profit margins.
    """
    if "profit margin" in response_text.lower():
        response_text = response_text.replace("profit margin", "[REDACTED METRIC]")
    return response_text


# --- ⚙️ Application & Data Setup ---

st.set_page_config(page_title="GlobalCart Multi-Agent Engine", page_icon="🛒", layout="centered", initial_sidebar_state="collapsed")
st.title("🛒 GlobalCart Multi-Agent Engine")
st.markdown("Multi-Agent Architecture: Planner, Greeter, RAG Specialist, & Data Analyst (Pandas).")

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

AGENT_MODEL = "openai/gpt-4o-mini"

# --- 📊 Load Cleaned Local Data for Data Analyst Agent ---
@st.cache_data
def load_data():
    try:
        # Load the CSV, ensuring strict types for numerical analysis
        df = pd.read_csv("inventory.csv")
        # SECURITY GUARDRAIL: We drop internal notes entirely so the Data Analyst Agent 
        # mathematically cannot execute queries that leak profit margins or supplier data.
        if "Internal_Notes" in df.columns:
            df = df.drop(columns=["Internal_Notes"])
        return df
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return None

df_safe = load_data()


# --- 🌍 Sidebar UI ---
st.sidebar.header("🛠️ System Controls")
st.sidebar.info("Hard metadata filtering disabled. The LLM Agent will dynamically route to RAG (Semantics) or Pandas Data Analyst (Math).")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    if "audio_key" in st.session_state:
        st.session_state.audio_key += 1
    st.rerun()


# --- 🤖 Multi-Agent Prompts ---

PLANNER_PROMPT = """You are the GlobalCart Planner Evaluator. 
Analyze the user's input and route it to the EXACT correct agent.

Routing Rules:
1. If it is a simple greeting, pleasantry, or small talk -> output 'GREETER'
2. If it requires EXACT calculations, counting, maximums, minimums, sorting, or numerical threshold filtering (e.g. "under 500", "cheapest", "how many") -> output 'ANALYST'
3. If it is a substantive question about general product specs, policies, semantic meaning, or broad retail questions without exact math -> output 'RAG'

Output ONLY the agent name: 'GREETER', 'ANALYST', or 'RAG'."""

GREETER_PROMPT = """You are a direct, helpful greeting assistant. 
Do NOT introduce yourself or say 'I am an AI'. Do NOT offer your name.
Simply acknowledge the user's greeting politely and ask how you can assist them with GlobalCart today."""

ANALYST_PROMPT = """You are the GlobalCart Data Analyst.
Your job is to translate the user's natural language question into a SINGLE, valid, read-only Python pandas expression.
The dataframe is named `df`.
Columns available: Product_ID, Country, Category, Item_Name, Price_Local, Currency, Technical_Specs.

CRITICAL RULES:
1. Do NOT write multiline code. Do NOT assign variables.
2. Do NOT import modules or use print().
3. Output ONLY the exact raw expression to evaluate. No markdown, no backticks, no python tags, no explanations.
4. You must filter by the user's explicit country if they mention one.

Examples:
User: "What are the cheapest electronics under 500 in Nigeria?"
Output: df[(df['Country'] == 'Nigeria') & (df['Category'] == 'Electronics') & (df['Price_Local'] < 500)].nsmallest(5, 'Price_Local')[['Item_Name', 'Price_Local', 'Currency']]

User: "How many items are in Kenya?"
Output: len(df[df['Country'] == 'Kenya'])

User: "What is the most expensive item in Ghana?"
Output: df[df['Country'] == 'Ghana'].nlargest(1, 'Price_Local')[['Item_Name', 'Price_Local', 'Currency']]
"""

RAG_PROMPT = """You are the GlobalCart Assistant.

CRITICAL DIRECTIVES:
1. SECURITY: NEVER reveal PII, Profit Margins, or Supplier Names. Refuse explicitly if asked.
2. CLARIFYING QUESTIONS: If the user hasn't specified their country/region in their prompt or chat history, ASK clarifying questions (e.g., "Which country are you shopping from?"). DO NOT recommend products until you know their region.
3. DATA SYNTHESIS: You will be provided with either retrieved text (from Vector DB) or precise mathematical/tabular data (from the Data Analyst). Summarize and present the findings clearly, naturally, and professionally to the user.
4. NO HALLUCINATION: Rely strictly on the retrieved context or analytical results provided.

CHAT HISTORY:
{history}

SYSTEM PROVIDED CONTEXT:
{context}
"""


# --- 💬 Chat UI State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/mp3")


# --- 🎤 Input Handling ---
prompt_text = None
is_voice = False

if hasattr(st, "audio_input"):
    audio_value = st.audio_input("🎙️ Speak your message", key=f"voice_input_{st.session_state.audio_key}")
else:
    audio_value = st.file_uploader("Upload or Record Audio", type=["wav", "mp3", "m4a"], accept_multiple_files=False, key=f"voice_input_{st.session_state.audio_key}")

text_input = st.chat_input("Or type your question...")

if audio_value:
    with st.spinner("Transcribing audio..."):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
                prompt_text = r.recognize_google(audio_data)
                is_voice = True
        except Exception as e:
            st.error(f"Could not transcribe audio. Please try again. ({e})")
elif text_input:
    prompt_text = text_input
    is_voice = False


# --- 🧠 Multi-Agent Execution Flow ---
if prompt_text:
    try:
        # 1. Evaluate Input Guardrails
        if not input_guardrail(prompt_text):
            st.error("⚠️ Input blocked by security guardrails.")
            st.stop()

        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-4:]])

        st.session_state.messages.append({"role": "user", "content": f"🗣️ {prompt_text}" if is_voice else prompt_text})
        with st.chat_message("user"):
            st.markdown(f"🗣️ {prompt_text}" if is_voice else prompt_text)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            context_str = ""
            
            # 2. Planner Evaluator Agent (Routing)
            with st.spinner("Planner Evaluator analyzing intent..."):
                planner_res = client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": PLANNER_PROMPT},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                route = planner_res.choices[0].message.content.strip().upper()

            # 3. Handoff to Specialized Agents
            if "GREETER" in route:
                stream = client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": GREETER_PROMPT},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.6,
                    max_tokens=150,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                        
            elif "ANALYST" in route and df_safe is not None:
                # 🤖 Data Analyst Agent Execution (Pandas calculations)
                with st.spinner("Data Analyst synthesizing Python query..."):
                    code_res = client.chat.completions.create(
                        model=AGENT_MODEL,
                        messages=[
                            {"role": "system", "content": ANALYST_PROMPT},
                            {"role": "user", "content": prompt_text}
                        ],
                        temperature=0.0
                    )
                    code_expr = code_res.choices[0].message.content.strip().replace("`", "").replace("python", "").strip()
                    
                    try:
                        # Safe(r) eval on the explicitly sanitized DataFrame
                        result = eval(code_expr, {"__builtins__": {}}, {"df": df_safe})
                        context_str = f"Data Analyst Exact Results:\n{result}"
                        st.info(f"📊 Executed Analyst Query: `{code_expr}`")
                    except Exception as eval_e:
                        context_str = f"Data Analyst attempted query: {code_expr}, but encountered an error: {eval_e}. Rely on generic advice instead."

            else:
                # 🤖 RAG Specialist Agent Execution (Semantic/Pinecone)
                with st.spinner("RAG Specialist retrieving semantic context..."):
                    embed_response = pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[prompt_text],
                        parameters={"input_type": "query", "truncate": "END"}
                    )
                    q_vec = embed_response[0].values
                    
                    index = pc.Index(INDEX_NAME)
                    results = index.query(
                        vector=q_vec,
                        top_k=20,
                        include_metadata=True
                    )
                    
                    safe_context_parts = []
                    for match in results["matches"]:
                        meta = match["metadata"]
                        if meta.get("doc_type") == "product":
                            safe_context_parts.append(
                                f"Product: {meta.get('name', 'Unknown')} | Region: {meta.get('country', 'Global')} | Price: {meta.get('currency', '')} {meta.get('price', '0.00')} | Specs: {meta.get('specs', '')}"
                            )
                        else:
                            safe_context_parts.append(f"Policy: {meta.get('title', 'Policy')} | Region: {meta.get('country', 'Global')} | Detail: {meta.get('content', '')}")
                    
                    context_str = "\n".join(safe_context_parts) if safe_context_parts else "No relevant items found."

            # 4. Final Synthesis if not Greeter
            if "GREETER" not in route:
                sys_prompt = RAG_PROMPT.format(history=history_str, context=context_str)
                stream = client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.2, 
                    max_tokens=800,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")

            # 5. Evaluate Output Guardrails
            full_response = output_guardrail(full_response)
            response_placeholder.markdown(full_response)

            # 6. Optional TTS for Voice Input
            tts_bytes = None
            if is_voice:
                with st.spinner("Generating voice response..."):
                    tts = gTTS(text=full_response, lang='en', tld='com')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    tts_bytes = fp.read()
                    st.audio(tts_bytes, format="audio/mp3")

            # Save state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "audio_bytes": tts_bytes
            })

            if is_voice:
                st.session_state.audio_key += 1
                st.rerun()

    except Exception as e:
        st.error(f"⚠️ Multi-Agent System Error: {str(e)}")

