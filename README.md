<div align="center">
  <h1>🛒 GlobalCart Intelligence Engine</h1>
  <p><b>An Advanced, Secure RAG System for International Retail Scale</b></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-black.svg?logo=pinecone)](https://www.pinecone.io/)
  [![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://langchain.com/)
  [![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
  
  <br />
</div>

## 📌 Overview

The **GlobalCart Intelligence Engine** is a production-grade Retrieval-Augmented Generation (RAG) architecture built to solve complex constraints in large-scale international retail. 

Querying massive, unstructured retail databases often results in hallucinations, cross-regional data contamination, and internal data leaks. This engine mitigates those risks through **Hard Metadata Filtering**, **Hybrid Search**, and **Implicit Data Masking**.

## 🏗️ Core Architecture & Guardrails

1. **Strict Regional Integrity (No Cross-Contamination)**  
   When a user queries the database from a specific region (e.g., *Ghana*), the system applies a hard filter to the Pinecone vector database (`filter={"country": {"$eq": country_code}}`). It is mathematically impossible for the retrieval engine to return prices, policies, or products meant for a different operational region.
   
2. **Implicit Data Masking (Red Team Tested)**  
   Retail datasets contain sensitive PII (Personally Identifiable Information), supplier contacts, and internal profit margins. This system implements a retrieval-layer guardrail that explicitly intercepts and strips out the `Internal_Notes` column from the vector context *before* it is passed to the LLM. 
   
3. **Zero Local Overhead**  
   The application is fully abstracted to the cloud to prevent memory exhaustion and allow serverless deployment:
   - **Embeddings:** Processed natively via the `pinecone.inference.embed` API (`multilingual-e5-large` model).
   - **Generation:** Handled entirely via the OpenRouter API (`meta-llama/llama-3.1-8b-instruct`).

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/eskayML/globalcart-intelligence-engine.git
cd globalcart-intelligence-engine
```

### 2. Install Dependencies
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Duplicate the `.env.example` file and add your API keys:
```bash
cp .env.example .env
```
Inside `.env`, populate:
```ini
PINECONE_API_KEY="your_pinecone_api_key_here"
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 4. Seed the Vector Database
Before launching the UI, you must index the raw `inventory.csv` into Pinecone. Run the seeding script once:
```bash
python seed_database.py
```

### 5. Launch the Application
Run the Streamlit frontend locally:
```bash
streamlit run app.py
```

---

## 🤝 Contributing
Contributions are welcome. Please ensure any pull requests involving retrieval mechanics do not bypass the core metadata filters or security guardrails.

## 📝 License
This project is licensed under the [MIT License](LICENSE).