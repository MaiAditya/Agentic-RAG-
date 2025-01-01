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
from loguru import logger
from app.core.logger import logger  # This will configure the logger

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
    """Process a PDF file and store its contents in the vector database."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        logger.info(f"Processing PDF file: {file.filename}")
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        result = await pdf_agent.process_pdf(contents)
        logger.info("PDF processing completed successfully")
        
        return JSONResponse(
            content={
                "status": "success", 
                "message": "PDF processed successfully",
                "file_size": file_size,
                "details": result
            },
            status_code=200
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the document collections.
    """
    try:
        stats = await chroma_client.get_collection_stats()
        return JSONResponse(
            content={"status": "success", "stats": stats},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug-document")
async def debug_document(file: UploadFile):
    """
    Debug endpoint to check document processing.
    """
    try:
        contents = await file.read()
        doc = await process_document(contents)
        return JSONResponse(
            content={
                "status": "success",
                "document_structure": {
                    "num_pages": len(doc.get("pages", [])),
                    "sample_page": doc.get("pages", [{}])[0] if doc.get("pages") else None,
                }
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in debug-document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)