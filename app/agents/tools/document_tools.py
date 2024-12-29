from .base import PDFTool
from typing import Dict, Any
from pydantic import Field

class DocumentSearchTool(PDFTool):
    name = "document_search"
    description: str = "Search for relevant information in the document"
    chroma_client: Any = Field(description="ChromaDB client for document search")
    
    def __init__(self, chroma_client):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation during initialization
        object.__setattr__(self, "chroma_client", chroma_client)
        
    def _run(self, query: str) -> Dict[str, Any]:
        """Synchronous implementation - calls async version"""
        raise NotImplementedError("Use async version of this tool")
        
    async def _arun(self, query: str) -> Dict[str, Any]:
        results = await self.chroma_client.query(query, limit=3)
        return {
            "type": "text",
            "content": results,
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