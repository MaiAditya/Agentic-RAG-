import streamlit as st
from utils.api_client import APIClient

class ResultsViewer:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def render(self):
        st.header("Document Statistics")
        
        # Get latest stats
        stats = self.api_client.get_stats()
        
        if stats.get("status") == "success":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Documents",
                    stats["stats"]["total_documents"]
                )
            
            with col2:
                collections = stats["stats"]["collections"]
                st.metric(
                    "Text Segments",
                    collections.get("text", {}).get("count", 0)
                )
            
            with col3:
                st.metric(
                    "Tables & Images",
                    collections.get("tables", {}).get("count", 0) +
                    collections.get("images", {}).get("count", 0)
                )
            
            # Detailed stats
            with st.expander("Detailed Statistics"):
                st.json(stats["stats"]) 