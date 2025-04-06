from fastapi import FastAPI
from document_api import router as document_router
from search_api import router as search_router
import os
from dotenv import load_dotenv

load_dotenv()

# Read from environment
APP_TITLE = os.getenv("APP_TITLE", "Graph RAG API")
APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Graph RAG API combining document and search services")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# Include routers from document_api and search_api
app.include_router(document_router)
app.include_router(search_router)

@app.get("/")
async def root():
    return {"message": "GraphRAG API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
