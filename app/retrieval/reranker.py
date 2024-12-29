from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import torch
from loguru import logger

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            logger.error(f"Error initializing reranker model: {e}")
            self.model = None

    def rerank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank the retrieved results using cross-encoder."""
        if not self.model or not results:
            return results

        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, result.get("content", "")) for result in results]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
                # Combine original score with rerank score
                result["score"] = 0.5 * result.get("score", 0) + 0.5 * result["rerank_score"]
            
            # Sort by combined score
            reranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            return reranked_results

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results

    def batch_rerank(self, results: List[Dict[str, Any]], query: str, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Rerank results in batches to handle memory constraints."""
        if not self.model or not results:
            return results

        try:
            all_pairs = [(query, result.get("content", "")) for result in results]
            all_scores = []

            # Process in batches
            for i in range(0, len(all_pairs), batch_size):
                batch_pairs = all_pairs[i:i + batch_size]
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores)

            # Update results with scores
            for result, score in zip(results, all_scores):
                result["rerank_score"] = float(score)
                result["score"] = 0.5 * result.get("score", 0) + 0.5 * result["rerank_score"]

            return sorted(results, key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Error during batch reranking: {e}")
            return results 