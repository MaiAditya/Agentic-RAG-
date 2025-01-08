import streamlit as st
from utils.api_client import APIClient

class PDFUploader:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def render(self):
        st.header("Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to analyze"
        )
        return uploaded_file
    
    def process_pdf(self, file) -> bool:
        try:
            response = self.api_client.process_pdf(file)
            if response.get("status") == "success":
                st.success("PDF processed successfully!")
                st.json(response["collection_stats"])
                return True
            else:
                st.error(f"Error processing PDF: {response.get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False 