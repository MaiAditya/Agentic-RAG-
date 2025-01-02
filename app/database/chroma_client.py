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
                if not docs:  # Skip if no documents for this type
                    continue
                    
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
                    cleaned_ids = []
                    
                    for doc in docs:
                        # Skip documents without required fields
                        if not all(key in doc for key in ["id", "content", "metadata"]):
                            logger.warning(f"Skipping document missing required fields: {doc}")
                            continue
                            
                        metadata = doc.get("metadata", {})
                        # Ensure all metadata values are strings
                        cleaned_metadata.append({
                            k: str(v) if v is not None else ""
                            for k, v in metadata.items()
                        })
                        
                        # Handle different content types
                        if doc_type == "images" and isinstance(doc["content"], bytes):
                            content = base64.b64encode(doc["content"]).decode('utf-8')
                        elif isinstance(doc["content"], (dict, list)):
                            content = json.dumps(doc["content"])
                        else:
                            content = str(doc["content"])
                        
                        cleaned_contents.append(content)
                        cleaned_ids.append(str(doc["id"]))
                    
                    if cleaned_contents:  # Only add if we have valid documents
                        collection.add(
                            documents=cleaned_contents,
                            ids=cleaned_ids,
                            metadatas=cleaned_metadata
                        )
                        logger.info(f"Added {len(cleaned_contents)} documents to {doc_type} collection")

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def query_collection(self, collection_name: str, query_text: str, 
                             min_similarity: float = 0.3, **kwargs):
        logger.info(f"""
        Querying ChromaDB collection:
        - Collection: {collection_name}
        - Query: {query_text}
        - Min similarity: {min_similarity}
        - Additional params: {kwargs}
        """)
        
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.warning(f"Collection {collection_name} not found")
                return None
            
            # Check if collection is empty
            if collection.count() == 0:
                logger.warning(f"Collection {collection_name} is empty")
                return None
            
            # Extract n_results from kwargs if present, otherwise use default
            n_results = kwargs.pop('n_results', 10)
            
            # Extract where clause if present
            where = kwargs.pop('where', None)
            
            query_params = {
                "query_texts": [query_text],
                "n_results": n_results
            }
            
            # Add where clause if present
            if where:
                query_params["where"] = where
            
            # Add any remaining kwargs
            query_params.update(kwargs)
            
            results = collection.query(**query_params)
            
            # Check if results are empty
            if not results or not any(results["documents"]):
                logger.info(f"No results found in collection {collection_name}")
                return None
            
            logger.info(f"Query results from {collection_name}: {json.dumps(results, indent=2)}")
            return results
            
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}", exc_info=True)
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections."""
        try:
            stats = {}
            for name, collection in self.collections.items():
                try:
                    count = collection.count()
                    stats[name] = {
                        "count": count,
                        "exists": True
                    }
                except Exception as e:
                    logger.error(f"Error getting stats for collection {name}: {e}")
                    stats[name] = {
                        "count": 0,
                        "exists": False,
                        "error": str(e)
                    }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise

