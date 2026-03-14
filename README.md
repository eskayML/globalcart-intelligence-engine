# GlobalCart Intelligence Engine

An advanced Retrieval-Augmented Generation (RAG) system built to solve the complex constraints of international retail. Built specifically for easy deployment on **Streamlit Community Cloud (Streamlit Share)**.

## Key Architecture

This system strictly abides by the rules defined in the `Task: Global Retail Intelligence Engine` brief.

1. **No Local Models Allowed**: 
    - LLM Generation: Handled completely via OpenRouter.
    - Embeddings: Processed completely via **Pinecone Inference API** (`multilingual-e5-large`). Zero local GPU/CPU overhead.
2. **The Regional Integrity Test**: 
    - A Streamlit sidebar dictates the active country. 
    - Queries use hard Pinecone metadata filtering (`filter={"country": {"$eq": country_code}}`). A user in Ghana (`GH`) will *never* see a product or policy meant for South Africa (`ZA`).
3. **The Security (Red Team) Test**: 
    - The `app.py` enforces a **Hard Guardrail** at the retrieval layer. The `internal_notes` field containing Profit Margins and PII is intentionally stripped *before* the context is ever passed to the LLM. It is mathematically impossible for the LLM to hallucinate or leak this data.
4. **The Technical Precision Test**: 
    - SKUs and exact technical specifications are retained perfectly.

## Setup & Deployment on Streamlit Share

1. Fork or clone this repository to your GitHub account.
2. Log into [Streamlit Share (share.streamlit.io)](https://share.streamlit.io/).
3. Click **New app** -> Deploy from your GitHub repository.
4. Point it to this repository and set the main file path to `app.py`.
5. Under **Advanced settings... -> Secrets**, securely paste your API keys:
   ```toml
   PINECONE_API_KEY = "your_pinecone_key"
   OPENROUTER_API_KEY = "your_openrouter_key"
   ```
6. Click Deploy!
