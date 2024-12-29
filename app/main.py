from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import nltk
import logging
import os
import shutil
import json
import asyncio
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_nltk():
    """Initialize NLTK with all required resources."""
    required_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'universal_tagset',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package)
            logger.info(f"Successfully downloaded NLTK package: {package}")
        except Exception as e:
            logger.warning(f"Error downloading NLTK package {package}: {e}")

# Initialize NLTK
initialize_nltk()

# Import your application components
from app.core.pdf_processor import PDFProcessor
from app.database.chroma_client import ChromaDBClient
from app.agents.pdf_agent import PDFAgent
from app.config import settings

app = FastAPI(
    title="Multimodal PDF Analysis Pipeline",
    description="An end-to-end pipeline for multimodal analysis of PDFs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
chroma_client = ChromaDBClient()
pdf_agent = PDFAgent(pdf_processor, chroma_client)

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """
    Process a PDF file and store its contents in the vector database.
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        logger.info(f"Processing PDF file: {file.filename}")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        result = await pdf_agent.process_pdf(contents)
        logger.info("PDF processing completed successfully")
        
        return JSONResponse(
            content={"status": "success", "message": "PDF processed successfully", "details": result},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: str):
    """
    Query the processed PDF contents.
    """
    try:
        response = await pdf_agent.answer_query(query)
        return JSONResponse(
            content={"status": "success", "response": response},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)