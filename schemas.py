from pydantic import BaseModel, Field
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    country: str = Field(description="The user's ISO-2 country code (e.g., 'GH', 'ZA', 'IN', 'NL', 'KE')")

class Product(BaseModel):
    id: str
    sku: str
    name: str
    description: str
    price: float
    currency: str
    country: str
    internal_notes: Optional[str] = None # PII/Margins (MUST NOT BE REVEALED)

class Policy(BaseModel):
    id: str
    title: str
    content: str
    country: str
    doc_type: str = "policy"