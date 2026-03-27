import streamlit as st
import os
import time
import io
from openai import OpenAI
from pinecone import Pinecone

try:
    import speech_recognition as sr
    from gtts import gTTS
except ImportError:
    st.error("Please run: pip install SpeechRecognition gTTS pydub")
    st.stop()

# --- 🛡️ Guardrail Definitions ---

def input_guardrail(prompt_text: str) -> bool:
    """
    STUB: Input Guardrail Evaluator.
    Intended to intercept malicious prompts, prompt injections, or prohibited queries 
    before they reach any agent.
    Returns False if the input is deemed unsafe.
    """
    restricted_keywords = ["ignore previous instructions", "system prompt", "bypass"]
    if any(keyword in prompt_text.lower() for keyword in restricted_keywords):
        return False
    return True

def output_guardrail(response_text: str) -> str:
    """
    STUB: Output Guardrail Filter.
    Intended to scan the generated agent response for PII, supplier leaks, 
    or internal profit margins before displaying it to the user.
    Returns a sanitized string.
    """
    # Example placeholder logic:
    if "profit margin" in response_text.lower():
        response_text = response_text.replace("profit margin", "[REDACTED METRIC]")
    return response_text


# --- ⚙️ Application & Key Setup ---

st.set_page_config(page_title="GlobalCart Multi-Agent Engine", page_icon="🛒", layout="centered", initial_sidebar_state="expanded")
st.title("🛒 GlobalCart Multi-Agent Engine")
st.markdown("Advanced Multi-Agent RAG System with Planner Evaluator & Streaming.")

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

# Initialize Clients (Using Official OpenAI SDK mapped to OpenRouter)
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Utilizing an OpenAI model explicitly via OpenRouter
AGENT_MODEL = "openai/gpt-4o-mini"


# --- 🌍 Sidebar UI ---
st.sidebar.header("🌍 Regional Context")
country_selection = st.sidebar.selectbox(
    "Select your shopping region:",
    ["Nigeria", "Ghana", "Kenya", "South Africa", "Netherlands", "Cameroon", "India", "Ivory Coast", "Rwanda", "Uganda"]
)
st.sidebar.info(f"Metadata Filtering active. You are locked into the **{country_selection}** catalog.")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_audio = None
    st.rerun()


# --- 🤖 Multi-Agent Prompts ---

PLANNER_PROMPT = """You are the GlobalCart Planner Evaluator. 
Analyze the user's input. If it is a simple greeting, pleasantry, or small talk, output EXACTLY the word 'GREETER'. 
If it is a substantive question about products, prices, policies, or retail, output EXACTLY the word 'RAG'.
Output nothing else."""

GREETER_PROMPT = """You are a direct, helpful greeting assistant. 
Do NOT introduce yourself or say 'I am an AI'. Do NOT offer your name.
Simply acknowledge the user's greeting politely and ask how you can assist them with GlobalCart today."""

RAG_PROMPT = """You are the GlobalCart RAG Specialist Agent.

CRITICAL DIRECTIVES:
1. SECURITY: NEVER reveal PII, Profit Margins, or Supplier Names. Refuse explicitly if asked.
2. REGIONAL INTEGRITY: Your context is pre-filtered for {country}. Only answer using the currency/rules provided.
3. CLARIFYING QUESTIONS: If the user's query is vague, broad, or lacks specifics (e.g. "show me laptops"), DO NOT answer immediately. Instead, ask 2 or 3 clarifying questions to narrow down their exact need (e.g. budget, brand preference, use case).
4. NO HALLUCINATION: Rely strictly on the retrieved context.

RETRIEVED CONTEXT:
{context}
"""


# --- 💬 Chat UI State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/mp3")


# --- 🎤 Input Handling ---
prompt_text = None
is_voice = False

if hasattr(st, "audio_input"):
    audio_value = st.audio_input("Speak to GlobalCart Agents")
else:
    audio_value = st.file_uploader("Upload or Record Audio", type=["wav", "mp3", "m4a"], accept_multiple_files=False)

text_input = st.chat_input("Or type your question...")

if audio_value and audio_value != st.session_state.last_audio:
    st.session_state.last_audio = audio_value
    with st.spinner("Transcribing audio..."):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
                prompt_text = r.recognize_google(audio_data)
                is_voice = True
        except Exception as e:
            st.error(f"Could not transcribe audio: {e}")
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

        # Render User Input
        st.session_state.messages.append({"role": "user", "content": f"🗣️ {prompt_text}" if is_voice else prompt_text})
        with st.chat_message("user"):
            st.markdown(f"🗣️ {prompt_text}" if is_voice else prompt_text)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # 2. Planner Evaluator Agent (Routing)
            with st.spinner("Planner Evaluator analyzing intent..."):
                planner_res = client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": PLANNER_PROMPT},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.0,
                    max_tokens=10,
                    top_p=1.0
                )
                route = planner_res.choices[0].message.content.strip().upper()

            # 3. Handoff to Specialized Agents
            if "GREETER" in route:
                # 🤖 Greeter Agent Execution
                stream = client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": GREETER_PROMPT},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.6,
                    max_tokens=150,
                    top_p=0.9,
                    stream=True  # Streaming front-facing model calls
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                        
            else:
                # 🤖 RAG Specialist Agent Execution
                with st.spinner("RAG Agent retrieving regional context..."):
                    embed_response = pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=[prompt_text],
                        parameters={"input_type": "query", "truncate": "END"}
                    )
                    q_vec = embed_response[0].values
                    
                    index = pc.Index(INDEX_NAME)
                    results = index.query(
                        vector=q_vec,
                        top_k=5,
                        include_metadata=True,
                        filter={"country": {"$eq": country_selection}}
                    )
                    
                    safe_context_parts = []
                    for match in results["matches"]:
                        meta = match["metadata"]
                        if meta.get("doc_type") == "product":
                            safe_context_parts.append(
                                f"Product: {meta.get('name', 'Unknown')} | Price: {meta.get('currency', '')} {meta.get('price', '0.00')} | Specs: {meta.get('specs', '')}"
                            )
                        else:
                            safe_context_parts.append(f"Policy: {meta.get('title', 'Policy')} | Detail: {meta.get('content', '')}")
                    
                    context_str = "\n".join(safe_context_parts) if safe_context_parts else "No relevant products found in this region."

                # Stream RAG Response
                sys_prompt = RAG_PROMPT.format(country=country_selection, context=context_str)
                stream = client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=0.2, # Low temperature for factual RAG
                    max_tokens=800,
                    top_p=0.9,
                    stream=True  # Streaming front-facing model calls
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")

            # 4. Evaluate Output Guardrails
            full_response = output_guardrail(full_response)
            response_placeholder.markdown(full_response)

            # 5. Optional TTS for Voice Input
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

    except Exception as e:
        # Exception handling as requested
        st.error(f"⚠️ Multi-Agent System Error: {str(e)}")

