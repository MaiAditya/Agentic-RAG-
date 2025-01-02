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
import uuid
import json

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
                    page_num,
                    doc
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

    def _process_page_sync(self, page: fitz.Page, page_num: int, doc: fitz.Document) -> Dict[str, Any]:
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
                        self.image_processor.process(page, doc),
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

    async def process(self, content: bytes) -> Dict[str, Any]:
        """Process a PDF file and extract its contents."""
        try:
            # Load PDF
            doc = fitz.open(stream=content, filetype="pdf")
            logger.info(f"Opened PDF with {len(doc)} pages")
            
            result = {
                "pages": [],
                "metadata": doc.metadata
            }
            
            # Process each page
            for page_num, page in enumerate(doc):
                logger.info(f"Processing page {page_num + 1}")
                page_content = {
                    "page": page_num,
                    "text": page.get_text(),
                    "images": [],
                    "tables": []
                }
                
                # Extract text blocks
                text_blocks = page.get_text("blocks")
                if text_blocks:
                    logger.info(f"Found {len(text_blocks)} text blocks on page {page_num + 1}")
                    page_content["text"] = "\n".join([block[4] for block in text_blocks if isinstance(block[4], str)])
                
                # Extract images if present
                images = await self.image_processor.process(page, doc)
                if images:
                    logger.info(f"Found {len(images)} images on page {page_num + 1}")
                    page_content["images"] = images
                
                # Extract tables if present
                tables = await self.table_processor.process(page)
                if tables:
                    logger.info(f"Found {len(tables)} tables on page {page_num + 1}")
                    page_content["tables"] = tables
                
                result["pages"].append(page_content)
            
            # Log summary in a single line
            logger.debug("Extracted content summary: " + json.dumps({
                'page_count': len(result['pages']),
                'has_text': any(page.get('text') for page in result['pages']),
                'total_images': sum(len(page.get('images', [])) for page in result['pages']),
                'total_tables': sum(len(page.get('tables', [])) for page in result['pages'])
            }))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

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

    async def process_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process extracted content into documents for vector storage."""
        documents = []
        
        try:
            # Process each page's content
            for page in content.get("pages", []):
                page_num = page.get("page", 0)
                
                # Process text content
                if page.get("text"):
                    text_doc = {
                        "id": f"text_{uuid.uuid4()}",
                        "type": "text",
                        "content": page["text"],
                        "metadata": {
                            "source": "text",
                            "page": page_num
                        }
                    }
                    documents.append(text_doc)
                    logger.info(f"Created text document for page {page_num}")
                
                # Process images
                for idx, img in enumerate(page.get("images", [])):
                    if img.get("caption"):
                        image_doc = {
                            "id": f"image_{uuid.uuid4()}",
                            "type": "images",
                            "content": img["caption"],
                            "metadata": {
                                "source": "image",
                                "page": page_num,
                                "index": idx,
                                "position": img.get("position", {})
                            }
                        }
                        documents.append(image_doc)
                        logger.info(f"Created image document for page {page_num}, image {idx}")
                
                # Process tables
                for idx, table in enumerate(page.get("tables", [])):
                    if table.get("content"):
                        table_doc = {
                            "id": f"table_{uuid.uuid4()}",
                            "type": "tables",
                            "content": table["content"],
                            "metadata": {
                                "source": "table",
                                "page": page_num,
                                "index": idx,
                                "position": table.get("position", {})
                            }
                        }
                        documents.append(table_doc)
                        logger.info(f"Created table document for page {page_num}, table {idx}")

            logger.info(f"Processed {len(documents)} total documents")
            return documents

        except Exception as e:
            logger.error(f"Error processing content into documents: {str(e)}")
            raise