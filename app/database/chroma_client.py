from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
import base64
import json

class ChromaDBClient:
    def __init__(self, persist_directory: str = "./chroma_db"):
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            
            # Use SentenceTransformer embedding function
            self.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Initialize collections with embedding function
            self.collections = {
                "text": self.client.get_or_create_collection(
                    "text", 
                    embedding_function=self.embedding_function
                ),
                "tables": self.client.get_or_create_collection(
                    "tables",
                    embedding_function=self.embedding_function
                ),
                "images": self.client.get_or_create_collection(
                    "images",
                    embedding_function=self.embedding_function
                )
            }
            logger.info(f"Successfully initialized ChromaDB with collections: {list(self.collections.keys())}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        try:
            by_type = {}
            for doc in documents:
                doc_type = doc.get("type", "text")
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(doc)

            for doc_type, docs in by_type.items():
                if doc_type == "images":
                    logger.info(f"""
Processing {len(docs)} image documents:
{json.dumps([{
    'id': doc['id'],
    'caption': doc.get('content', ''),
    'metadata': doc.get('metadata', {}),
    'page': doc.get('metadata', {}).get('page', 'unknown'),
    'position': doc.get('metadata', {}).get('position', 'unknown')
} for doc in docs], indent=2)}
""")
                if doc_type in self.collections:
                    collection = self.collections[doc_type]
                    cleaned_metadata = []
                    cleaned_contents = []
                    
                    for doc in docs:
                        metadata = doc.get("metadata", {})
                        cleaned_metadata.append({
                            k: str(v) if v is not None else ""
                            for k, v in metadata.items()
                        })
                        
                        # Handle image content differently
                        if doc_type == "images" and isinstance(doc["content"], bytes):
                            # Convert bytes to base64 string for storage
                            content = base64.b64encode(doc["content"]).decode('utf-8')
                        else:
                            content = doc["content"]
                        cleaned_contents.append(content)
                    
                    collection.add(
                        documents=cleaned_contents,
                        ids=[str(doc["id"]) for doc in docs],
                        metadatas=cleaned_metadata
                    )
                    logger.info(f"Added {len(docs)} documents to {doc_type} collection")

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def query_collection(self, collection_name: str, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query a specific collection.
        
        Args:
            collection_name: Name of the collection to query
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Dict containing query results
        """
        try:
            if collection_name not in self.collections:
                logger.warning(f"Collection {collection_name} not found")
                return None
            
            collection = self.collections[collection_name]
            
            # Perform the query
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            if results and len(results['documents']) > 0:
                return {
                    "documents": results['documents'][0],  # First query's results
                    "metadatas": results['metadatas'][0],
                    "distances": results['distances'][0],
                    "ids": results['ids'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error querying {collection_name} collection: {str(e)}")
            raise

