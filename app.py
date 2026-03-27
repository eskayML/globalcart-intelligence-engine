import streamlit as st
import os
import time
import io
try:
    import speech_recognition as sr
    from gtts import gTTS
except ImportError:
    st.error("Please run: pip install SpeechRecognition gTTS pydub")
    st.stop()

from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Configure Streamlit UI
st.set_page_config(page_title="GlobalCart AI Engine", page_icon="🛒", layout="centered", initial_sidebar_state="expanded")

st.title("🛒 GlobalCart Intelligence Engine")
st.markdown("Advanced Retail RAG System with Strict Regional & Security Guardrails.")

# Robust API Key Fetcher
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

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0.0
)

# --- Sidebar: Regional Context & Reset ---
st.sidebar.header("🌍 Regional Context")
country_selection = st.sidebar.selectbox(
    "Select your shopping region:",
    ["Nigeria", "Ghana", "Kenya", "South Africa", "Netherlands", "Cameroon", "India", "Ivory Coast", "Rwanda", "Uganda"]
)
st.sidebar.info(f"Metadata Filtering active. You are physically locked into the **{country_selection}** catalog.")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_audio = None
    st.rerun()

SYS_PROMPT = """You are the GlobalCart AI Assistant. 
You provide extremely precise, helpful answers based strictly on the provided context and prior conversation.

CRITICAL DIRECTIVES:
1. SECURITY: You MUST NEVER reveal PII (Names, Emails, Phone Numbers), Profit Margins, or Supplier Names. If a user asks for internal or private data, explicitly refuse to answer.
2. REGIONAL INTEGRITY: The context is pre-filtered for the user's country. Only answer using the currency and rules in the context.
3. CONVERSATIONAL MEMORY: Use the 'Chat History' below to answer follow-up questions seamlessly.
4. NO HALLUCINATION: Never invent a price or specification.

USER COUNTRY: {country}

CHAT HISTORY:
{history}

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""

# --- Chat UI State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/mp3")
        if msg.get("context_data"):
            with st.expander("🔍 View Retrieved Sources"):
                for src in msg["context_data"]:
                    st.code(src, language="json")

# --- Input Handling (Voice & Text) ---
prompt_text = None
is_voice = False

# Fallback for older Streamlit versions without st.audio_input
if hasattr(st, "audio_input"):
    audio_value = st.audio_input("Speak to GlobalCart Assistant")
else:
    audio_value = st.file_uploader("Upload or Record Audio", type=["wav", "mp3", "m4a"], accept_multiple_files=False)

text_input = st.chat_input("Or type your question...")

if audio_value and audio_value != st.session_state.last_audio:
    st.session_state.last_audio = audio_value
    with st.spinner("Transcribing audio..."):
        try:
            r = sr.Recognizer()
            # Convert uploaded/recorded audio to AudioFile
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
                prompt_text = r.recognize_google(audio_data)
                is_voice = True
        except Exception as e:
            st.error(f"Could not transcribe audio: {e}")

elif text_input:
    prompt_text = text_input
    is_voice = False

# --- RAG Execution ---
if prompt_text:
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-4:]])
    
    # Show user prompt
    st.session_state.messages.append({"role": "user", "content": f"🗣️ {prompt_text}" if is_voice else prompt_text})
    with st.chat_message("user"):
        st.markdown(f"🗣️ {prompt_text}" if is_voice else prompt_text)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Executing secure retrieval & generation..."):
            try:
                # 1. Embed query
                embed_response = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[prompt_text],
                    parameters={"input_type": "query", "truncate": "END"}
                )
                q_vec = embed_response[0].values
                
                # 2. Hard Metadata Filter on Retrieval
                index = pc.Index(INDEX_NAME)
                results = index.query(
                    vector=q_vec,
                    top_k=5,
                    include_metadata=True,
                    filter={"country": {"$eq": country_selection}}
                )
                
                # 3. Construct Safe Context
                safe_context_parts = []
                raw_sources = []
                for match in results["matches"]:
                    meta = match["metadata"]
                    clean_meta = {k: v for k, v in meta.items() if k.lower() != "internal_notes"}
                    raw_sources.append(f"Score: {match['score']:.2f}\n{clean_meta}")
                    
                    if meta.get("doc_type") == "product":
                        safe_context_parts.append(
                            f"Product: {meta.get('name', 'Unknown')} | ID: {meta.get('product_id', 'Unknown')} | "
                            f"Category: {meta.get('category', 'Unknown')} | "
                            f"Price: {meta.get('currency', '')} {meta.get('price', '0.00')} | Specs: {meta.get('specs', '')}"
                        )
                    else:
                        safe_context_parts.append(f"Policy: {meta.get('title', 'Policy')} | Detail: {meta.get('content', '')}")
                        
                context_str = "\n".join(safe_context_parts)
                
                if not context_str:
                    final_response = f"I could not find any relevant information for your query in the **{country_selection}** region catalog."
                else:
                    # 4. Generate Answer
                    prompt_template = PromptTemplate.from_template(SYS_PROMPT)
                    chain = prompt_template | llm
                    
                    resp = chain.invoke({
                        "country": country_selection,
                        "history": history_str,
                        "context": context_str,
                        "question": prompt_text
                    })
                    final_response = resp.content
                
                st.markdown(final_response)
                
                # Generate TTS if voice was used
                tts_bytes = None
                if is_voice:
                    with st.spinner("Generating voice response..."):
                        tts = gTTS(text=final_response, lang='en', tld='com')
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        tts_bytes = fp.read()
                        st.audio(tts_bytes, format="audio/mp3")

                if raw_sources:
                    with st.expander("🔍 View Retrieved Sources"):
                        for src in raw_sources:
                            st.code(src, language="json")
                
                # Save to state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_response,
                    "audio_bytes": tts_bytes,
                    "context_data": raw_sources if raw_sources else None
                })
                
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
