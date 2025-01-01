from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
            collection_results = await self.chroma_client.query_collection(
                collection,
                query,
                limit=limit
            )
            
            if collection_results:  # Only process if we got results
                # Apply collection-specific weights
                weighted_results = self._apply_weights(collection_results, collection)
                results.extend(weighted_results)
        
        if not results:
            logger.warning(f"No results found in any collection for query: {query}")
            return []
            
        # Sort by score and return top results
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]
    
    def _apply_weights(self, results: List[Dict], collection: str) -> List[Dict]:
        weight = self.weights.get(collection, 1.0)
        for result in results:
            result["score"] = result.get("score", 0) * weight
        return results 