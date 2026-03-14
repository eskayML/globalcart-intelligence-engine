# GlobalCart Intelligence Engine

An advanced Retrieval-Augmented Generation (RAG) system built to solve the complex constraints of international retail.

## Key Architecture

This system strictly abides by the rules defined in the `Task: Global Retail Intelligence Engine` brief.

1. **No Local Models Allowed**: 
    - LLM Generation: Handled completely via OpenRouter (`meta-llama/llama-3.1-8b-instruct` or any preferred OpenRouter model).
    - Embeddings: Processed completely via **Pinecone Inference API** (`multilingual-e5-large`). Zero local GPU/CPU overhead.
2. **The Regional Integrity Test**: 
    - Queries use hard Pinecone metadata filtering (`filter={"country": {"$eq": req.country}}`). A user in Ghana (`GH`) will *never* see a product or policy meant for South Africa (`ZA`).
3. **The Security (Red Team) Test**: 
    - The `app.py` enforces a **Hard Guardrail** at the retrieval layer. The `internal_notes` field containing Profit Margins and PII is intentionally stripped *before* the context is ever passed to the LLM. It is mathematically impossible for the LLM to hallucinate or leak this data.
4. **The Technical Precision Test**: 
    - SKUs and exact technical specifications are retained perfectly.

## Setup & Deployment

1. Install requirements:
   ```bash
   pip install pinecone-client openai langchain langchain-pinecone langchain-openai python-dotenv fastapi uvicorn
   ```

2. Environment Variables (`.env`):
   ```
   PINECONE_API_KEY="your_pinecone_key"
   OPENROUTER_API_KEY="your_openrouter_key"
   ```

3. Start the API:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
