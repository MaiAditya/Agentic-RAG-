from typing import List, Dict, Any
import fitz  # PyMuPDF
import os
import asyncio
from loguru import logger
from .text_processor import TextProcessor
from .table_processor import TableProcessor
from .image_processor import ImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class PDFProcessor:
    def __init__(self):
        # Initialize specialized processors
        self.text_processor = TextProcessor()
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor()
        # Use CPU count for optimal parallelization
        self.max_workers = max(multiprocessing.cpu_count() - 1, 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process a PDF file from disk using parallel processing."""
        try:
            doc = fitz.open(file_path)
            metadata = await self._extract_metadata(doc)
            
            # Create a list to store futures
            futures = []
            loop = asyncio.get_event_loop()
            
            # Submit all pages for processing
            for page_num in range(len(doc)):
                future = loop.run_in_executor(
                    self.executor,
                    self._process_page_sync,
                    doc[page_num],
                    page_num
                )
                futures.append(future)
            
            # Process all pages in parallel and collect results
            pages = []
            for completed_future in await asyncio.gather(*futures):
                if completed_future:
                    pages.append(completed_future)
            
            # Sort pages by page number to maintain order
            pages.sort(key=lambda x: x['page_num'])
            
            return {
                "pages": pages,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return None
        finally:
            if 'doc' in locals():
                doc.close()

    def _process_page_sync(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Process a single page synchronously."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Process all elements concurrently within this thread
                text_blocks, tables, images, layout = loop.run_until_complete(
                    asyncio.gather(
                        self.text_processor.process(page),
                        self.table_processor.process(page),
                        self.image_processor.process(page),
                        self._analyze_layout(page)
                    )
                )
                
                return {
                    "page_num": page_num,
                    "text_blocks": text_blocks,
                    "tables": tables,
                    "images": images,
                    "layout": layout
                }
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return {
                "page_num": page_num,
                "text_blocks": [],
                "tables": [],
                "images": [],
                "layout": {}
            }

    async def process(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process PDF content using parallel processing."""
        temp_path = "temp.pdf"
        doc = None
        try:
            with open(temp_path, "wb") as f:
                f.write(pdf_content)

            doc = fitz.open(temp_path)
            metadata = await self._extract_metadata(doc)
            
            # Create futures for parallel processing
            futures = []
            loop = asyncio.get_event_loop()
            
            # Submit all pages for processing
            for page_num in range(len(doc)):
                future = loop.run_in_executor(
                    self.executor,
                    self._process_page_sync,
                    doc[page_num],
                    page_num
                )
                futures.append(future)
            
            # Process all pages in parallel and collect results
            pages = []
            for completed_future in await asyncio.gather(*futures):
                if completed_future:
                    pages.append(completed_future)
            
            # Sort pages by page number
            pages.sort(key=lambda x: x['page_num'])
            
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

    async def _analyze_layout(self, page: fitz.Page) -> Dict[str, Any]:
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

    async def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract PDF document metadata."""
        try:
            # Get basic metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "page_count": len(doc)
            }
            
            # Get file size safely
            try:
                metadata["file_size"] = os.path.getsize(doc.name)
            except (AttributeError, OSError):
                metadata["file_size"] = 0

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}