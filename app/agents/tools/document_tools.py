from .base import PDFTool
from typing import Dict, Any
from pydantic import Field
from loguru import logger

class DocumentSearchTool(PDFTool):
    name = "document_search"
    description: str = "Search for relevant information in the document using keywords or phrases"
    chroma_client: Any = Field(description="ChromaDB client for document search")
    
    def __init__(self, chroma_client):
        super().__init__()
        object.__setattr__(self, "chroma_client", chroma_client)
        
    def _run(self, query: str) -> Dict[str, Any]:
        """Synchronous implementation - calls async version"""
        raise NotImplementedError("Use async version of this tool")
        
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Search across all collections for relevant information."""
        try:
            logger.info(f"Executing document search with query: {query}")
            results = []
            collections = ["text", "tables", "images"]
            
            for collection in collections:
                try:
                    collection_results = await self.chroma_client.query_collection(
                        collection_name=collection,
                        query_text=query,
                        n_results=3
                    )
                    
                    if collection_results and isinstance(collection_results, dict):
                        # Extract documents and metadata
                        documents = collection_results.get("documents", [[]])[0]
                        metadatas = collection_results.get("metadatas", [[]])[0]
                        distances = collection_results.get("distances", [[]])[0]
                        
                        # Process each document with its metadata and distance
                        for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
                            result_item = {
                                "content": doc,
                                "metadata": meta,
                                "source": collection,
                                "score": 1.0 - (distances[idx] if idx < len(distances) else 0)
                            }
                            results.append(result_item)
                        
                        logger.info(f"Found {len(documents)} results in {collection}")
                        
                except Exception as e:
                    logger.warning(f"Error searching {collection} collection: {str(e)}")
                    continue
            
            logger.info(f"Total results found: {len(results)}")
            return {
                "type": "text",
                "content": results,
                "source": "document_search"
            }
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return {
                "type": "text",
                "content": [],
                "source": "document_search"
            }

class TableSearchTool(PDFTool):
    name = "table_search"
    description: str = "Search for relevant tables in the document"
    chroma_client: Any = Field(description="ChromaDB client for table search")
    
    def __init__(self, chroma_client):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation during initialization
        object.__setattr__(self, "chroma_client", chroma_client)
        
    def _run(self, query: str) -> Dict[str, Any]:
        """Synchronous implementation - calls async version"""
        raise NotImplementedError("Use async version of this tool")
        
    async def _arun(self, query: str) -> Dict[str, Any]:
        results = await self.chroma_client.query_collection("tables", query, limit=2)
        return {
            "type": "table",
            "content": results,
            "source": "table_search"
        } 

class ImageSearchTool(PDFTool):
    name = "image_search"
    description: str = "Search for relevant images using natural language descriptions"
    chroma_client: Any = Field(description="ChromaDB client for image search")
    
    def __init__(self, chroma_client):
        super().__init__()
        object.__setattr__(self, "chroma_client", chroma_client)
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Synchronous implementation - calls async version"""
        raise NotImplementedError("Use async version of this tool")
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Search for relevant images based on the query"""
        results = await self.chroma_client.query_collection("images", query, limit=3)
        
        # Process results to include image descriptions and metadata
        processed_results = []
        for result in results:
            processed_results.append({
                "description": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("score", 0),
                "image_type": result.get("metadata", {}).get("image_type", "unknown"),
                "position": result.get("metadata", {}).get("position", {})
            })
        
        return {
            "type": "image",
            "content": processed_results,
            "source": "image_search"
        }