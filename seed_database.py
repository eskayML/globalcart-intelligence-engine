import os
import csv
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

# 2. Load dataset from CSV
def load_csv(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return data
        
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Typecast price safely
            if row.get('price'):
                try:
                    row['price'] = float(row['price'])
                except ValueError:
                    row['price'] = 0.0
            data.append(row)
    return data

csv_file = "inventory.csv"
inventory_data = load_csv(csv_file)

if not inventory_data:
    print("No data to process. Exiting.")
    exit()

def create_embedding_text(item):
    """Creates the text chunk that Pinecone will actually embed and search against."""
    if item.get("doc_type") == "product":
        # Include SKU, Name, and Desc to ensure hybrid/semantic search catches it
        return f"Product: {item.get('name', '')}. SKU: {item.get('sku', '')}. Description: {item.get('description', '')}"
    else:
        return f"Policy: {item.get('title', '')}. Content: {item.get('content', '')}"

print(f"Preparing to embed and index {len(inventory_data)} items from {csv_file}...")

# 3. Batch process and upload
BATCH_SIZE = 50 # Safe batch size for Pinecone Inference API
for i in range(0, len(inventory_data), BATCH_SIZE):
    batch = inventory_data[i:i+BATCH_SIZE]
    
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
                "doc_type": item.get("doc_type", ""),
                "country": item.get("country", ""),
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