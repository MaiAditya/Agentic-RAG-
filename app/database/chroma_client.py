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
import uuid
from collections import defaultdict

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        """Initialize the embedding model with explicit CPU configuration."""
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
        
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"  # Explicitly use CPU
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for the input texts."""
        try:
            embeddings = self.model.encode(
                input,
                convert_to_tensor=False,  # Return numpy array instead of torch tensor
                normalize_embeddings=True  # L2 normalize embeddings
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

class ChromaDBClient:
    def __init__(self):
        """Initialize ChromaDB client and collections."""
        try:
            # Initialize embedding function
            embedding_function = LocalEmbeddingFunction()
            
            # Initialize ChromaDB client with new configuration format
            self.client = chromadb.Client(
                chromadb.Settings(
                    persist_directory="./chroma_db",
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )

            # Create or get collections for different content types
            self.collections = {
                "text": self.client.get_or_create_collection(
                    name="text",
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                ),
                "tables": self.client.get_or_create_collection(
                    name="tables",
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                ),
                "images": self.client.get_or_create_collection(
                    name="images",
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            }
            logger.info(f"Successfully initialized ChromaDB with collections: {list(self.collections.keys())}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

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
        try:
            # Group documents by type
            grouped_docs = defaultdict(list)
            for doc in documents:
                grouped_docs[doc["type"]].append(doc)

            # Process text documents
            if "text" in grouped_docs:
                text_docs = grouped_docs["text"]
                await self._add_to_collection(
                    "text",
                    [doc["id"] for doc in text_docs],
                    [doc["content"] for doc in text_docs],
                    [doc["metadata"] for doc in text_docs]
                )

            # Process table documents
            if "tables" in grouped_docs:
                table_docs = grouped_docs["tables"]
                await self._add_to_collection(
                    "tables",
                    [doc["id"] for doc in table_docs],
                    # Convert table content to string if it's a dict
                    [doc["content"]["raw_text"] if isinstance(doc["content"], dict) else str(doc["content"]) 
                     for doc in table_docs],
                    [doc["metadata"] for doc in table_docs]
                )

            # Process image documents
            if "images" in grouped_docs:
                image_docs = grouped_docs["images"]
                await self._add_to_collection(
                    "images",
                    [doc["id"] for doc in image_docs],
                    [doc["content"] for doc in image_docs],
                    [doc["metadata"] for doc in image_docs]
                )

        except Exception as e:
            logger.error(f"Error adding documents to {doc['type']}: {str(e)}")
            logger.error(f"Document structure: {documents}")
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

    async def query_collection(self, collection_name: str, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query a specific collection.
        
        Args:
            collection_name: Name of collection to query
            query_text: The query text
            n_results: Number of results to return
            
        Returns:
            Dict containing query results
        """
        try:
            collection = self.collections[collection_name]
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            return {
                "documents": results["documents"][0],  # First query results
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0]
            }
            
        except Exception as e:
            logger.error(f"Error querying {collection_name} collection: {str(e)}")
            return None

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

    async def verify_collections(self) -> Dict[str, Any]:
        """Verify the health of all collections."""
        verification = {}
        for name, collection in self.collections.items():
            try:
                # Test adding a document
                test_id = f"test_{uuid.uuid4()}"
                collection.add(
                    documents=["Test document"],
                    metadatas=[{"source": "verification"}],
                    ids=[test_id]
                )
                
                # Test querying
                results = collection.query(
                    query_texts=["test"],
                    n_results=1
                )
                
                # Clean up test document
                collection.delete(ids=[test_id])
                
                verification[name] = {
                    "status": "healthy",
                    "count": collection.count(),
                    "can_add": True,
                    "can_query": bool(results["documents"])
                }
            except Exception as e:
                verification[name] = {
                    "status": "error",
                    "error": str(e),
                    "count": 0,
                    "can_add": False,
                    "can_query": False
                }
        
        return verification

    async def _add_to_collection(self, collection_name: str, ids: List[str], contents: List[str], 
                               metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to a specific collection.
        
        Args:
            collection_name: Name of the collection to add documents to
            ids: List of document IDs
            contents: List of document contents
            metadatas: List of document metadata
        """
        try:
            # Get the collection
            collection = self.collections[collection_name]
            
            # Process metadata to ensure compatibility with ChromaDB
            processed_metadatas = [self._prepare_metadata(m) for m in metadatas]
            
            # Add documents to collection
            collection.add(
                documents=contents,
                ids=ids,
                metadatas=processed_metadatas
            )
            
            logger.info(f"Successfully added {len(ids)} documents to {collection_name} collection")
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {str(e)}")
            raise

