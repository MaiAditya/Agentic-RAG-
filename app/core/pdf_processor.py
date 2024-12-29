from typing import List, Dict, Any
import fitz  # PyMuPDF
import os
from .text_processor import TextProcessor
from .table_processor import TableProcessor
from .image_processor import ImageProcessor
from loguru import logger

class PDFProcessor:
    def __init__(self):
        # Initialize specialized processors
        self.text_processor = TextProcessor()
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor()

    async def process(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process PDF content using specialized processors for text, tables, and images."""
        temp_path = "temp.pdf"
        doc = None
        try:
            with open(temp_path, "wb") as f:
                f.write(pdf_content)

            doc = fitz.open(temp_path)
            pages = []
            metadata = self._extract_metadata(doc)  # Extract metadata before processing pages

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                text_blocks = await self.text_processor.process(page)
                tables = await self.table_processor.process(page)
                images = await self.image_processor.process(page)
                
                pages.append({
                    "page_num": page_num,
                    "text_blocks": text_blocks,
                    "tables": tables,
                    "images": images,
                    "layout": self._analyze_layout(page)
                })

            return {
                "pages": pages,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
        finally:
            if doc:
                doc.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _analyze_layout(self, page: fitz.Page) -> Dict[str, Any]:
        """Analyze the layout of a page."""
        try:
            return {
                "width": page.rect.width,
                "height": page.rect.height,
                "rotation": page.rotation,
                "mediabox": list(page.mediabox),
                "cropbox": list(page.cropbox)
            }
        except Exception as e:
            logger.error(f"Error analyzing layout: {e}")
            return {
                "width": 0,
                "height": 0,
                "rotation": 0,
                "mediabox": [0, 0, 0, 0],
                "cropbox": [0, 0, 0, 0]
            }

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract PDF document metadata."""
        try:
            return {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "page_count": len(doc),
                "file_size": doc.filesize
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}