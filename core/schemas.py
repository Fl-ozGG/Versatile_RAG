from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., example="¿Sobre que trata el documento cargado?")
    openai_key: str
    pinecone_key: str
    index_name: str
    top_k: Optional[int] = 3  # Cuántos documentos recuperar por defecto