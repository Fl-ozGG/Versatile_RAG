from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from core.schemas import QueryRequest, ConfigRequest
from core.engine import RAGEngine
import shutil
import os


app = FastAPI()

rag_instances = {}

@app.post("/configure")
async def configure_rag(config: ConfigRequest):
    try:
        engine = RAGEngine(
            openai_key=config.openai_key,
            pinecone_key=config.pinecone_key,
            index_name=config.index_name
        )
        
        rag_instances["default"] = engine
        
        return {"status": "Configuración guardada correctamente"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error de configuración: {str(e)}")

@app.post("/ask")
async def ask_question(question: str):
    engine = rag_instances.get("default")
    if not engine:
        raise HTTPException(status_code=400, detail="El RAG no está configurado. Llama primero a /configure")
    
    try:
        answer = engine.ask(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    engine = rag_instances.get("default")
    if not engine:
        raise HTTPException(status_code=400, detail="Configura el RAG primero en /configure")

    # Creamos una ruta temporal para guardar el archivo y procesarlo
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        num_chunks = engine.ingest_file(temp_path)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": num_chunks,
            "message": "Documento indexado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Importante: Borrar el archivo temporal después de procesarlo
        if os.path.exists(temp_path):
            os.remove(temp_path)