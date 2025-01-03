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
from app.language_models.llm import create_llm

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

# Global components
pdf_agent = None
chroma_client = None

async def init_components():
    """Initialize all components needed for the application."""
    global chroma_client
    
    pdf_processor = PDFProcessor()
    chroma_client = ChromaDBClient()
    llm = await create_llm()
    
    pdf_agent = PDFAgent(
        pdf_processor=pdf_processor,
        chroma_client=chroma_client
    )
    return pdf_agent

@app.on_event("startup")
async def startup_event():
    global pdf_agent
    pdf_agent = await init_components()

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """Process a PDF file and store its contents in the vector database."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        logger.info(f"Processing PDF file: {file.filename}")
        contents = await file.read()
        file_size = len(contents)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        # Process the PDF
        result = await pdf_agent.process_pdf(contents)
        
        # Get updated collection stats
        stats = await chroma_client.get_collection_stats()
        
        return JSONResponse(
            content={
                "status": "success", 
                "message": "PDF processed successfully",
                "file_size": file_size,
                "details": result,
                "collection_stats": stats
            },
            status_code=200
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/query")
async def query(query: dict):
    """
    Query the processed PDF contents.
    """
    try:
        logger.info(f"Received query request: {json.dumps(query, indent=2)}")
        
        if not pdf_agent:
            raise HTTPException(
                status_code=500,
                detail="PDF Agent not properly initialized"
            )
            
        if not query or "query" not in query:
            logger.warning("Invalid query format - missing 'query' field")
            raise HTTPException(status_code=400, detail="Query must be provided in request body")
            
        query_text = query["query"].strip()
        if not query_text:
            logger.warning("Empty query text received")
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Extract optional filters
        filters = query.get("filters", {})
        min_similarity = filters.get("min_similarity", 0.3)
        content_types = filters.get("content_type", ["text", "tables", "images"])
        page = filters.get("page")
        
        logger.info(f"""
        Processing query with parameters:
        - Query text: {query_text}
        - Min similarity: {min_similarity}
        - Content types: {content_types}
        - Page filter: {page}
        """)
        
        # Get collection stats before query
        pre_query_stats = await chroma_client.get_collection_stats()
        logger.info(f"Collection stats before query: {json.dumps(pre_query_stats, indent=2)}")
        
        response = await pdf_agent.answer_query(
            query_text,
            min_similarity=min_similarity,
            content_types=content_types,
            page=page
        )
        
        if not response:
            logger.warning(f"""
            No results found for query:
            - Query text: {query_text}
            - Content types searched: {content_types}
            - Min similarity threshold: {min_similarity}
            """)
            return JSONResponse(
                content={
                    "status": "no_results",
                    "message": "No relevant information found",
                    "response": None,
                    "debug_info": {
                        "collection_stats": pre_query_stats,
                        "query_parameters": {
                            "text": query_text,
                            "filters": filters
                        }
                    }
                },
                status_code=200
            )
            
        logger.info(f"Query successful - Found relevant information: {json.dumps(response, indent=2)}")
        return JSONResponse(
            content={
                "status": "success", 
                "response": response,
                "query": query_text,
                "metadata": {
                    "filters_applied": filters,
                    "content_types_searched": content_types,
                    "collection_stats": pre_query_stats
                }
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        logger.error(f"Query details: {json.dumps(query, indent=2)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check ChromaDB collections
        collection_stats = await chroma_client.get_collection_stats()
        
        # Check if collections exist and have documents
        collections_health = {
            "text": collection_stats.get("text", {}).get("count", 0) > 0,
            "tables": collection_stats.get("tables", {}).get("count", 0) > 0,
            "images": collection_stats.get("images", {}).get("count", 0) > 0
        }
        
        return JSONResponse(
            content={
                "status": "healthy",
                "collections": collections_health,
                "chroma_db": collection_stats
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get detailed statistics about document collections."""
    try:
        stats = await chroma_client.get_collection_stats()
        
        # Add more detailed statistics
        detailed_stats = {
            "collections": {},
            "total_documents": 0,
            "health": {
                "all_collections_exist": True,
                "total_collections": len(stats)
            }
        }
        
        for collection_name in ["text", "tables", "images"]:
            collection_info = stats.get(collection_name, {})
            doc_count = collection_info.get("count", 0)
            exists = collection_info.get("exists", False)
            
            detailed_stats["collections"][collection_name] = {
                "count": doc_count,
                "has_documents": doc_count > 0,
                "exists": exists,
                "error": collection_info.get("error")
            }
            
            detailed_stats["total_documents"] += doc_count
            if not exists:
                detailed_stats["health"]["all_collections_exist"] = False
        
        return JSONResponse(
            content={
                "status": "success",
                "stats": detailed_stats
            },
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
