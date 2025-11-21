# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.ML_framework_classification.router import router as classification_router

app = FastAPI(
    title="AI Sales Intelligence + Dynamic ML Framework",
    description="Multi-module AI platform with Dynamic Classification Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classification_router, prefix="/api/classification", tags=["Dynamic ML Classification"])

@app.get("/")
async def root():
    return {"message": "Dynamic ML Classification Framework is running!", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")