from fastapi import FastAPI, HTTPException
from core.schemas import QueryRequest

app = FastAPI()

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    try:
        # 1. Aquí llamarías a tu lógica en engine.py
        engine.procesar_pregunta(request)
        
        # Simulamos la respuesta para probar que el endpoint funciona
        return {
            "answer": f"Procesando tu pregunta: '{request.question}' en el índice '{request.index_name}'",
            "context_used": request.top_k,
            "status": "success"
        }
    except Exception as e:
        # Si algo sale mal (ej. llaves inválidas), devolvemos un error 500
        raise HTTPException(status_code=500, detail=str(e))