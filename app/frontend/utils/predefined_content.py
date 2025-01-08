from typing import Dict, List, Optional
import os
import json
from pathlib import Path
import shutil

class PredefinedContent:
    def __init__(self):
        self.pdfs_dir = Path("additional_pdfs")
        self.predefined_queries_file = self.pdfs_dir / "predefined_queries.json"
        
        # Create directory if it doesn't exist
        if not self.pdfs_dir.exists():
            self.pdfs_dir.mkdir(parents=True)
            
        # Create default predefined queries file if it doesn't exist
        if not self.predefined_queries_file.exists():
            self._create_default_queries()
            
        self.predefined_queries = self._load_predefined_queries()
    
    def _create_default_queries(self):
        """Create a default predefined queries file."""
        default_queries = {
            "sample.pdf": [
                "What are the main topics covered?",
                "Summarize the key findings",
                "List the recommendations"
            ]
        }
        with open(self.predefined_queries_file, 'w') as f:
            json.dump(default_queries, f, indent=2)
    
    def get_available_pdfs(self) -> List[str]:
        """Get list of available predefined PDFs."""
        if not self.pdfs_dir.exists():
            return []
        return [f.name for f in self.pdfs_dir.glob("*.pdf")]
    
    def get_pdf_path(self, pdf_name: str) -> Path:
        """Get full path for a predefined PDF."""
        return self.pdfs_dir / pdf_name
    
    def get_predefined_queries(self) -> Dict[str, List[str]]:
        """Get predefined queries for PDFs."""
        return self.predefined_queries
    
    def add_predefined_query(self, pdf_name: str, query: str) -> bool:
        """Add a new predefined query for a PDF."""
        try:
            if pdf_name not in self.predefined_queries:
                self.predefined_queries[pdf_name] = []
            
            if query not in self.predefined_queries[pdf_name]:
                self.predefined_queries[pdf_name].append(query)
                self._save_predefined_queries()
            return True
        except Exception:
            return False
    
    def upload_predefined_pdf(self, file) -> bool:
        """Upload a PDF to the predefined PDFs directory."""
        try:
            if not file.name.endswith('.pdf'):
                return False
                
            file_path = self.pdfs_dir / file.name
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file, f)
                
            # Initialize empty query list for new PDF
            if file.name not in self.predefined_queries:
                self.predefined_queries[file.name] = []
                self._save_predefined_queries()
                
            return True
        except Exception:
            return False
    
    def _load_predefined_queries(self) -> Dict[str, List[str]]:
        """Load predefined queries from JSON file."""
        if not self.predefined_queries_file.exists():
            return {}
        try:
            with open(self.predefined_queries_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_predefined_queries(self):
        """Save predefined queries to JSON file."""
        with open(self.predefined_queries_file, 'w') as f:
            json.dump(self.predefined_queries, f, indent=2) 