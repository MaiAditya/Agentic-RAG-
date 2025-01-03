from typing import List, Dict, Any
import numpy as np
import logging
import json
from loguru import logger

class HybridRetriever:
    def __init__(self, chroma_client):
        self.chroma_client = chroma_client
        self.collections = ["text", "images", "tables"]
        self.weights = {
            "text": 0.6,
            "images": 0.2,
            "tables": 0.2
        }
    
    async def get_relevant_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = []
        
        for collection in self.collections:
            try:
                collection_results = await self.chroma_client.query_collection(
                    collection,
                    query,
                    n_results=limit
                )
                
                if collection_results and collection_results.get("documents"):  # Check if we have documents
                    # Process and weight the results
                    processed_results = self._process_collection_results(
                        collection_results,
                        collection
                    )
                    results.extend(processed_results)
            except Exception as e:
                logger.error(f"Error querying collection {collection}: {str(e)}")
                continue
        
        if not results:
            logger.warning(f"No results found in any collection for query: {query}")
            return []
            
        # Sort by score and return top results
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]
    
    def _process_collection_results(self, results: Dict[str, Any], collection: str) -> List[Dict[str, Any]]:
        """Process and weight results from a collection."""
        processed_results = []
        weight = self.weights.get(collection, 1.0)
        
        # ChromaDB returns lists of lists for documents, metadatas, distances
        for doc, metadata, distance in zip(
            results["documents"][0],  # First list of documents
            results["metadatas"][0],  # First list of metadatas
            results["distances"][0]   # First list of distances
        ):
            # Convert distance to similarity score (closer to 1 is better)
            similarity_score = 1.0 / (1.0 + distance)
            
            # Create result dictionary
            result = {
                "content": doc,
                "metadata": metadata,
                "collection": collection,
                "score": similarity_score * weight  # Apply collection weight
            }
            
            # For tables and images, try to parse the content if it's a JSON string
            if collection in ["tables", "images"] and isinstance(doc, str):
                try:
                    result["content"] = json.loads(doc)
                except json.JSONDecodeError:
                    pass
                
            processed_results.append(result)
            
        return processed_results 