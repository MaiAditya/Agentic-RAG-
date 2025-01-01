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
import uuid

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
    try:
        stats = await chroma_client.get_collection_stats()
        return {
            "status": "healthy",
            "chroma_db": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        doc = await pdf_processor.process(contents)
        return JSONResponse(
            content={
                "status": "success",
                "document_structure": {
                    "num_pages": len(doc.get("pages", [])),
                    "sample_page": doc.get("pages", [{}])[0] if doc.get("pages") else None,
                    "metadata": doc.get("metadata", {}),
                    "content_types": {
                        "has_text": "text" in doc,
                        "has_images": bool(doc.get("images", [])),
                        "has_tables": bool(doc.get("tables", []))
                    }
                }
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error in debug-document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-add")
async def test_add_document():
    """Test endpoint to add a sample document to each collection"""
    try:
        test_docs = [
            {
                "id": f"test_text_{uuid.uuid4()}",
                "type": "text",
                "content": "This is a test text document.",
                "metadata": {"source": "test"}
            },
            {
                "id": f"test_image_{uuid.uuid4()}",
                "type": "images",
                "content": "A test image caption",
                "metadata": {"source": "test"}
            },
            {
                "id": f"test_table_{uuid.uuid4()}",
                "type": "tables",
                "content": "Test table content",
                "metadata": {"source": "test"}
            }
        ]
        
        await chroma_client.add_documents(test_docs)
        
        stats = await chroma_client.get_collection_stats()
        return {
            "status": "success",
            "message": "Test documents added",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error in test addition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection-info")
async def get_collection_info():
    """Get detailed information about collections and their contents."""
    try:
        collections_info = {}
        for name, collection in chroma_client.collections.items():
            try:
                count = collection.count()
                sample = None
                if count > 0:
                    sample = collection.peek()
                collections_info[name] = {
                    "count": count,
                    "sample": sample,
                    "exists": True
                }
            except Exception as e:
                collections_info[name] = {
                    "error": str(e),
                    "exists": False
                }
        
        return {
            "status": "success",
            "collections": collections_info
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-processing")
async def test_processing(file: UploadFile = File(...)):
    """Test endpoint to verify PDF processing pipeline."""
    try:
        # Read file content
        contents = await file.read()
        logger.info(f"Processing test file: {file.filename}")
        
        # Extract content
        extracted = await pdf_processor.process(contents)
        
        # Process into documents
        documents = []
        for page in extracted.get("pages", []):
            # Process text
            if page.get("text"):
                doc = {
                    "id": f"text_{uuid.uuid4()}",
                    "type": "text",
                    "content": page["text"],
                    "metadata": {"page": page["page"]}
                }
                documents.append(doc)
            
            # Process images
            for img in page.get("images", []):
                if img.get("caption"):
                    doc = {
                        "id": f"image_{uuid.uuid4()}",
                        "type": "images",
                        "content": img["caption"],
                        "metadata": {"page": page["page"]}
                    }
                    documents.append(doc)
            
            # Process tables
            for table in page.get("tables", []):
                if table.get("content"):
                    doc = {
                        "id": f"table_{uuid.uuid4()}",
                        "type": "tables",
                        "content": table["content"],
                        "metadata": {"page": page["page"]}
                    }
                    documents.append(doc)
        
        # Try adding to ChromaDB
        if documents:
            await chroma_client.add_documents(documents)
        
        return {
            "status": "success",
            "extraction": {
                "page_count": len(extracted["pages"]),
                "document_count": len(documents),
                "documents": documents[:2],  # Show first two documents as sample
            },
            "collection_stats": await chroma_client.get_collection_stats()
        }
        
    except Exception as e:
        logger.error(f"Error in test processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)