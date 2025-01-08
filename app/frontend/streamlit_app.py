import streamlit as st
from components.pdf_uploader import PDFUploader
from components.query_interface import QueryInterface
from components.results_viewer import ResultsViewer
from utils.api_client import APIClient
from utils.predefined_content import PredefinedContent

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Analysis Pipeline",
    page_icon="📄",
    layout="wide"
)

def main():
    # Initialize components
    api_client = APIClient()
    predefined_content = PredefinedContent()
    
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
    
    # Create tabs for different upload methods
    upload_tab, predefined_tab = st.tabs(["Upload New PDF", "Use Predefined PDF"])
    
    with upload_tab:
        # PDF Upload Section
        pdf_uploader = PDFUploader(api_client)
        uploaded_file = pdf_uploader.render()
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                success = pdf_uploader.process_pdf(uploaded_file)
                if success:
                    st.session_state.processed_pdf = True
                    st.session_state.collection_stats = api_client.get_stats()
    
    with predefined_tab:
        st.header("Predefined PDFs")
        
        # Display available predefined PDFs
        available_pdfs = predefined_content.get_available_pdfs()
        
        if not available_pdfs:
            st.info("No predefined PDFs available. Upload some using the section below.")
        else:
            selected_pdf = st.selectbox(
                "Select a predefined PDF",
                available_pdfs,
                help="Choose from previously uploaded PDFs"
            )
            
            if selected_pdf:
                if st.button("Load Selected PDF"):
                    pdf_path = predefined_content.get_pdf_path(selected_pdf)
                    with open(pdf_path, 'rb') as f:
                        with st.spinner("Processing predefined PDF..."):
                            success = pdf_uploader.process_pdf(f)
                            if success:
                                st.session_state.processed_pdf = True
                                st.session_state.collection_stats = api_client.get_stats()
        
        # Upload new predefined PDF
        st.markdown("---")
        st.subheader("Add New Predefined PDF")
        new_pdf = st.file_uploader(
            "Upload a PDF to add to predefined collection",
            type="pdf",
            key="predefined_pdf_uploader"
        )
        
        if new_pdf:
            if st.button("Add to Predefined PDFs"):
                if predefined_content.upload_predefined_pdf(new_pdf):
                    st.success(f"Successfully added {new_pdf.name} to predefined PDFs")
                    st.rerun()
                else:
                    st.error("Failed to add PDF to predefined collection")
    
    # Query Interface - Always show
    st.markdown("---")
    query_interface = QueryInterface(api_client)
    query_interface.render()
    
    # Results Viewer - Always show
    st.markdown("---")
    results_viewer = ResultsViewer(api_client)
    results_viewer.render()

if __name__ == "__main__":
    main() 