import streamlit as st
from components.pdf_uploader import PDFUploader
from components.query_interface import QueryInterface
from components.results_viewer import ResultsViewer
from utils.api_client import APIClient

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Analysis Pipeline",
    page_icon="📄",
    layout="wide"
)

def main():
    # Initialize API client
    api_client = APIClient()
    
    # Sidebar
    st.sidebar.title("PDF Analysis Pipeline")
    st.sidebar.markdown("---")
    
    # Check API health
    health_status = api_client.check_health()
    if health_status.get("status") == "healthy":
        st.sidebar.success("Backend API: Connected")
    else:
        st.sidebar.error("Backend API: Disconnected")
        st.error("Cannot connect to backend service. Please check if the server is running.")
        return

    # Main content
    st.title("PDF Document Analysis")
    
    # Initialize session state
    if 'processed_pdf' not in st.session_state:
        st.session_state.processed_pdf = False
    if 'collection_stats' not in st.session_state:
        st.session_state.collection_stats = None
    
    # PDF Upload Section
    pdf_uploader = PDFUploader(api_client)
    uploaded_file = pdf_uploader.render()
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            success = pdf_uploader.process_pdf(uploaded_file)
            if success:
                st.session_state.processed_pdf = True
                st.session_state.collection_stats = api_client.get_stats()
    
    # Query Interface
    if st.session_state.processed_pdf:
        st.markdown("---")
        query_interface = QueryInterface(api_client)
        query_interface.render()
    
    # Results Viewer
    if st.session_state.collection_stats:
        st.markdown("---")
        results_viewer = ResultsViewer(api_client)
        results_viewer.render()

if __name__ == "__main__":
    main() 