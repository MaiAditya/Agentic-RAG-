import requests
import streamlit as st
from typing import Optional, Dict, Any

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def check_health(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except:
            return {"status": "unhealthy"}
    
    def process_pdf(self, file) -> Dict[str, Any]:
        files = {"file": file}
        response = requests.post(f"{self.base_url}/process-pdf", files=files)
        return response.json()
    
    def query_pdf(self, query: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        data = {
            "query": query,
            "filters": filters or {}
        }
        response = requests.post(f"{self.base_url}/query", json=data)
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/stats")
        return response.json() 