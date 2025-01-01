from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings
import json
from loguru import logger
from .initialize_db import initialize_chroma

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class ChromaDBClient:
    def __init__(self):
        self.client, self.collections = initialize_chroma()
        # Keep default collection for backward compatibility
        self.collection = self.collections["text"]

    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """Convert complex metadata types to strings for ChromaDB compatibility."""
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                processed_metadata[key] = value
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                processed_metadata[key] = json.dumps(list(value))
            else:
                processed_metadata[key] = str(value)
        return processed_metadata

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the appropriate collections."""
        try:
            logger.info(f"Adding {len(documents)} documents to collections")
            
            for doc in documents:
                for page in doc.get("pages", []):
                    # Log counts for each type
                    text_blocks = len(page.get("text_blocks", []))
                    images = len(page.get("images", []))
                    tables = len(page.get("tables", []))
                    logger.info(f"Processing page {page['page_num']}: {text_blocks} text blocks, {images} images, {tables} tables")
                    
                    # Process text blocks
                    for block in page.get("text_blocks", []):
                        try:
                            metadata = {
                                "page_num": page["page_num"],
                                "block_id": block["id"],
                                "position": json.dumps(block["position"]),
                                "type": "text"
                            }
                            
                            self.collections["text"].add(
                                documents=[block["text"]],
                                metadatas=[self._prepare_metadata(metadata)],
                                ids=[f"text_{page['page_num']}_{block['id']}"]
                            )
                            logger.debug(f"Added text block: {block['text'][:100]}...")
                        except Exception as e:
                            logger.error(f"Error adding text block: {e}")
                    
                    # Similar logging for images and tables...
                    
            logger.info("Document addition completed")
            
        except Exception as e:
            logger.error(f"Error in add_documents: {e}")
            raise

    async def query(self, query_text: str, limit: int = 3) -> Dict[str, Any]:
        """Query all collections and return relevant results."""
        results = {}
        
        for collection_name, collection in self.collections.items():
            # Get the collection size
            collection_size = collection.count()
            # Adjust limit if needed
            actual_limit = min(limit, collection_size)
            
            if actual_limit > 0:
                query_results = collection.query(
                    query_texts=[query_text],
                    n_results=actual_limit
                )
                
                results[collection_name] = {
                    "documents": query_results["documents"][0],
                    "metadatas": [
                        {k: json.loads(v) if k in ["position", "size"] and isinstance(v, str) else v 
                         for k, v in m.items()}
                        for m in query_results["metadatas"][0]
                    ],
                    "distances": query_results["distances"][0]
                }
            else:
                results[collection_name] = {
                    "documents": [],
                    "metadatas": [],
                    "distances": []
                }
        
        return results

    async def query_collection(self, collection_name: str, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Query a specific collection."""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.error(f"Collection {collection_name} not found")
                return []

            collection_size = collection.count()
            actual_limit = min(limit, collection_size)
            
            if actual_limit == 0:
                return []

            results = collection.query(
                query_texts=[query_text],
                n_results=actual_limit
            )
            
            return [
                {
                    "content": doc,
                    "metadata": {
                        k: json.loads(v) if k in ["position", "size"] and isinstance(v, str) else v 
                        for k, v in meta.items()
                    },
                    "score": 1 - dist  # Convert distance to similarity score
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
            
        except Exception as e:
            logger.exception(f"Error querying collection {collection_name}: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed stats about collections."""
        stats = {}
        try:
            # Get list of all collections
            all_collections = self.client.list_collections()
            stats["available_collections"] = [c.name for c in all_collections]
            
            # Get counts for our specific collections
            collection_counts = {}
            for name, collection in self.collections.items():
                try:
                    count = collection.count()
                    peek = collection.peek() if count > 0 else None
                    collection_counts[name] = {
                        "count": count,
                        "sample": peek,
                        "exists": True
                    }
                except Exception as e:
                    logger.error(f"Error accessing collection {name}: {e}")
                    collection_counts[name] = {
                        "count": 0,
                        "error": str(e),
                        "exists": False
                    }
            
            stats["collections"] = collection_counts
            stats["persist_directory"] = self.client._settings.persist_directory
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            stats["error"] = str(e)
        
        return stats

