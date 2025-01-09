import requests
import streamlit as st
from typing import Optional, Dict, Any

class APIClient:
    def __init__(self, base_url: str = "http://localhost:4223"):
        self.base_url = base_url
    
    def check_health(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except:
            return {"status": "unhealthy"}
    
    def process_pdf(self, file) -> Dict[str, Any]:
        try:
            files = {"file": file}
            response = requests.post(f"{self.base_url}/process-pdf", files=files)
            return response.json()
        except requests.exceptions.JSONDecodeError:
            return {"status": "error", "detail": "Invalid response from server"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    
    def query_pdf(self, query: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            data = {
                "query": query,
                "filters": filters or {}
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(f"{self.base_url}/query", json=data, headers=headers)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "detail": f"Server returned status code {response.status_code}"
                }
            
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                return {
                    "status": "error",
                    "detail": "Invalid JSON response from server"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "detail": f"Request failed: {str(e)}"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/stats")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            return {"status": "error", "detail": "Invalid response from server"}
        except Exception as e:
            return {"status": "error", "detail": str(e)} 