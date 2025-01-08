import streamlit as st
from utils.api_client import APIClient

class QueryInterface:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def render(self):
        st.header("Query Document")
        
        # Query input
        query = st.text_input("Enter your question about the document")
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            with col1:
                min_similarity = st.slider(
                    "Minimum Similarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )
            with col2:
                content_types = st.multiselect(
                    "Content Types",
                    ["text", "tables", "images"],
                    default=["text", "tables", "images"]
                )
        
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing query..."):
                    filters = {
                        "min_similarity": min_similarity,
                        "content_type": content_types
                    }
                    response = self.api_client.query_pdf(query, filters)
                    
                    if response.get("status") == "success":
                        st.success("Query processed successfully!")
                        self._display_results(response)
                    else:
                        st.warning(response.get("message", "No relevant information found"))
            else:
                st.warning("Please enter a query")
    
    def _display_results(self, response):
        st.subheader("Results")
        
        # Display main response
        st.markdown("### Answer")
        st.write(response["response"])
        
        # Display metadata
        with st.expander("Query Details"):
            st.json(response["metadata"]) 