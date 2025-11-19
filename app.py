from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.db_routes import router as db_router

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="API for building and querying permanent RAG vector database"
)

# CORS (allow everything for now — adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(db_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG API running"}
