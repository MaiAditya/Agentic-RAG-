from .base import PDFTool
from typing import Dict, Any

class DocumentSearchTool(PDFTool):
    name = "document_search"
    description = "Search for relevant information in the document"
    
    def __init__(self, chroma_client):
        super().__init__()
        self.chroma_client = chroma_client
        
    async def _arun(self, query: str) -> Dict[str, Any]:
        results = await self.chroma_client.query(query, limit=3)
        return {
            "type": "text",
            "content": results,
            "source": "document_search"
        }

class TableSearchTool(PDFTool):
    name = "table_search"
    description = "Search for relevant tables in the document"
    
    def __init__(self, chroma_client):
        super().__init__()
        self.chroma_client = chroma_client
        
    async def _arun(self, query: str) -> Dict[str, Any]:
        results = await self.chroma_client.query_collection("tables", query, limit=2)
        return {
            "type": "table",
            "content": results,
            "source": "table_search"
        } 