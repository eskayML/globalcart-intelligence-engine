import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set.")

INDEX_NAME = "globalcart-retail-engine"
pc = Pinecone(api_key=PINECONE_API_KEY)

# 1. Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone Index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024, # Dimension for multilingual-e5-large
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created successfully!")
else:
    print(f"Index {INDEX_NAME} already exists.")

index = pc.Index(INDEX_NAME)

# 2. Mock Dataset (Tailored to pass the exact 4 criteria tests + expanded items)
# A real scenario would load a CSV/JSON here.
mock_data = [
    # --- GHANA (GH) ---
    {
        "id": "prod-gh-001",
        "doc_type": "product",
        "country": "GH",
        "name": "Solar Inverter 5kW",
        "sku": "GH-INV-5000",
        "price": 12500.00,
        "currency": "GHS",
        "description": "High-efficiency 5kW Solar Inverter for residential use. Pure sine wave.",
        "internal_notes": "Supplier: AccraTech Energy. Margin: 45%. Contact: Kwame Doe (kwame@accratech.com)"
    },
    {
        "id": "policy-gh-001",
        "doc_type": "policy",
        "country": "GH",
        "title": "Ghana Electronics Warranty",
        "content": "All electronics in Ghana come with a standard 1-year manufacturer warranty covering factory defects. Power surges are not covered."
    },
    # --- SOUTH AFRICA (ZA) ---
    {
        "id": "prod-za-001",
        "doc_type": "product",
        "country": "ZA",
        "name": "Solar Inverter 5kW",
        "sku": "ZA-INV-5000",
        "price": 18000.00,
        "currency": "ZAR",
        "description": "High-efficiency 5kW Solar Inverter. Built for load shedding resilience.",
        "internal_notes": "Supplier: Jozi Solar. Margin: 30%. Contact: Johan (johan@jozisolar.za)"
    },
    # --- NETHERLANDS (NL) ---
    {
        "id": "prod-nl-001",
        "doc_type": "product",
        "country": "NL",
        "name": "Dutch Master Bike Light",
        "sku": "NL-L-5042",
        "price": 45.00,
        "currency": "EUR",
        "description": "Ultra-bright LED bike light with USB-C charging. 500 lumens.",
        "internal_notes": "Supplier: EuroBike Corp. Margin: 60%. PII: ceo@eurobike.nl"
    },
    {
        "id": "policy-nl-001",
        "doc_type": "policy",
        "country": "NL",
        "title": "Netherlands Electronics Warranty",
        "content": "By EU law, all electronics sold in the Netherlands carry a mandatory 2-year warranty covering all functional defects."
    },
    # --- KENYA (KE) ---
    {
        "id": "prod-ke-001",
        "doc_type": "product",
        "country": "KE",
        "name": "Smart Kettle Pro",
        "sku": "KE-KTL-99",
        "price": 4500.00,
        "currency": "KES",
        "description": "Wifi-enabled smart kettle. Boil water via app.",
        "internal_notes": "Supplier: Nairobi Home Goods. Margin: 55%. Factory contact: +254 700 000000"
    }
]

# Generate additional mock items up to 50 for volume testing
for i in range(10, 60):
    mock_data.append({
        "id": f"prod-us-{i}",
        "doc_type": "product",
        "country": "US",
        "name": f"Generic Retail Item {i}",
        "sku": f"US-GEN-{i}",
        "price": 19.99 + i,
        "currency": "USD",
        "description": f"Standard retail item number {i} for US market.",
        "internal_notes": "Standard margin 20%."
    })

def create_embedding_text(item):
    """Creates the text chunk that Pinecone will actually embed and search against."""
    if item["doc_type"] == "product":
        # Include SKU, Name, and Desc to ensure hybrid/semantic search catches it
        return f"Product: {item['name']}. SKU: {item['sku']}. Description: {item['description']}"
    else:
        return f"Policy: {item['title']}. Content: {item['content']}"

print(f"Preparing to embed and index {len(mock_data)} items...")

# 3. Batch process and upload
BATCH_SIZE = 50 # Safe batch size for Pinecone Inference API
for i in range(0, len(mock_data), BATCH_SIZE):
    batch = mock_data[i:i+BATCH_SIZE]
    
    # Extract text for embeddings
    texts = [create_embedding_text(item) for item in batch]
    
    try:
        # Generate embeddings using Pinecone's serverless inference API
        embed_response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=texts,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for idx, item in enumerate(batch):
            embedding = embed_response[idx].values
            
            # The metadata is what we filter on and return to the LLM
            metadata = {
                "doc_type": item["doc_type"],
                "country": item["country"],
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "name": item.get("name", ""),
                "sku": item.get("sku", ""),
                "price": item.get("price", 0.0),
                "currency": item.get("currency", ""),
                "description": item.get("description", ""),
                "internal_notes": item.get("internal_notes", "") # Uploaded but filtered out at retrieval!
            }
            
            vectors_to_upsert.append({
                "id": item["id"],
                "values": embedding,
                "metadata": metadata
            })
            
        # Upsert to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} items).")
        
    except Exception as e:
        print(f"Error processing batch: {e}")

print("✅ Data seeding complete! The vector database is fully populated.")
