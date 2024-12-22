from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings
import json

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L12-v2',
            device=settings.EMBEDDING_DEVICE
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class ChromaDBClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Initialize embedding function
        self.embedding_function = LocalEmbeddingFunction()
        
        self.collections = {
            "text": self.client.get_or_create_collection(
                name="text_content",
                embedding_function=self.embedding_function
            ),
            "images": self.client.get_or_create_collection(
                name="image_content",
                embedding_function=self.embedding_function
            ),
            "tables": self.client.get_or_create_collection(
                name="table_content",
                embedding_function=self.embedding_function
            )
        }

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
        for doc in documents:
            for page in doc.get("pages", []):
                # Process text blocks
                for block in page.get("text_blocks", []):
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
                
                # Process images
                for img in page.get("images", []):
                    metadata = {
                        "page_num": page["page_num"],
                        "image_id": img["id"],
                        "position": json.dumps(img["position"]),
                        "size": json.dumps(img["size"]),
                        "format": img["format"],
                        "type": "image"
                    }
                    
                    self.collections["images"].add(
                        documents=[f"Image on page {page['page_num']}"],
                        metadatas=[self._prepare_metadata(metadata)],
                        ids=[f"image_{page['page_num']}_{img['id']}"]
                    )
                
                # Process tables
                for table in page.get("tables", []):
                    metadata = {
                        "page_num": page["page_num"],
                        "table_id": table["id"],
                        "position": json.dumps(table.get("position", {})),
                        "type": "table"
                    }
                    
                    self.collections["tables"].add(
                        documents=[json.dumps(table.get("content", []))],
                        metadatas=[self._prepare_metadata(metadata)],
                        ids=[f"table_{page['page_num']}_{table['id']}"]
                    )

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
