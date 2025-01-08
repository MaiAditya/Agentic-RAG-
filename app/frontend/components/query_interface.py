import streamlit as st
from utils.api_client import APIClient
from utils.predefined_content import PredefinedContent

class QueryInterface:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.predefined_content = PredefinedContent()
    
    def _process_query(self, query: str, min_similarity: float = 0.3, content_types: list = None):
        if not query:
            st.warning("Please enter a query")
            return
            
        with st.spinner("Processing query..."):
            filters = {
                "min_similarity": min_similarity,
                "content_types": content_types or ["text", "tables", "images"]
            }
            response = self.api_client.query_pdf(query, filters)
            
            if response.get("status") == "success":
                st.success("Query processed successfully!")
                st.write("Answer:", response.get("response", "No answer found"))
            elif response.get("status") == "no_results":
                st.warning("No relevant information found for your query")
            else:
                st.error(f"Error processing query: {response.get('detail', 'Unknown error')}")
                if response.get("detail"):
                    st.code(response["detail"])
    
    def render(self):
        st.header("Query Document")
        
        # Add tabs for query methods
        tab1, tab2 = st.tabs(["Custom Query", "Predefined Queries"])
        
        with tab1:
            # Original query input
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
            
            if st.button("Submit Custom Query"):
                self._process_query(query, min_similarity, content_types)
                
        with tab2:
            # Get predefined queries from JSON
            queries = {
                "Medical Queries": self.predefined_content.get_predefined_queries().get("medical_queries", []),
                "General Queries": self.predefined_content.get_predefined_queries().get("general_queries", [])
            }
            
            # Add document-specific queries if available
            available_pdfs = self.predefined_content.get_available_pdfs()
            for pdf in available_pdfs:
                pdf_queries = self.predefined_content.get_predefined_queries().get(pdf, [])
                if pdf_queries:
                    queries[f"Queries for {pdf}"] = pdf_queries
            
            if not queries:
                st.warning("No predefined queries available")
            else:
                # Category selection
                category = st.selectbox(
                    "Select Query Category",
                    list(queries.keys()),
                    help="Choose a category of predefined queries"
                )
                
                if category and queries[category]:
                    # Query selection
                    selected_query = st.selectbox(
                        "Select a Predefined Query",
                        queries[category],
                        help="Choose from predefined queries"
                    )
                    
                    if st.button("Run Predefined Query"):
                        self._process_query(selected_query)
                else:
                    st.info("No queries available for this category") 