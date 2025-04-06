from fastapi import FastAPI
from document_api import router as document_router
from search_api import router as search_router

app = FastAPI(
    title="Graph RAG API",
    description="Graph RAG API combining document and search services",
    version="1.0.0"
)

# Include routers from document_api and search_api
app.include_router(document_router)
app.include_router(search_router)

@app.get("/")
async def root():
    return {"message": "GraphRAG API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
