from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image
import fitz
from dataclasses import dataclass
import pandas as pd
from loguru import logger 
import torch
from transformers import TableTransformerForObjectDetection
from PIL import Image

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

@dataclass
class TableBoundary:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

class EnhancedTableProcessor:
    def __init__(self, min_confidence: float = 0.5):
        self.logger = logger
        self.min_confidence = min_confidence
        try:
            # Initialize Table Transformer model
            self.table_detector = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            # Initialize Table Structure Recognition model
            self.structure_recognizer = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.table_detector.to(self.device)
            self.structure_recognizer.to(self.device)
            
            # Initialize PaddleOCR for text extraction if available
            if PADDLE_AVAILABLE:
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
            else:
                self.logger.warning("PaddleOCR not available. Text extraction will be limited.")
                self.ocr_engine = None
            
            self.logger.info(f"Enhanced table processor initialized successfully with min_confidence={min_confidence}")
        except Exception as e:
            self.logger.error(f"Error initializing enhanced table processor: {str(e)}")
            raise
    async def process(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Process a PDF page to extract tables."""
        self.logger.info(f"Processing page {page.number} for table detection. Page size: {page.rect}")
        
        # Convert page to image for table detection
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        # Detect table boundaries
        table_boundaries = self._detect_tables(img_np)
        self.logger.info(f"Found {len(table_boundaries)} potential tables on page {page.number}")
        
        # Extract and process each detected table
        tables = []
        for idx, boundary in enumerate(table_boundaries):
            self.logger.info(
                f"Processing table {idx + 1}/{len(table_boundaries)} on page {page.number}:\n"
                f"Position: ({boundary.x0:.1f}, {boundary.y0:.1f}, {boundary.x1:.1f}, {boundary.y1:.1f})\n"
                f"Confidence: {boundary.confidence:.2f}"
            )
            
            table_content = self._extract_table_content(page, boundary)
            if table_content:
                if "error" in table_content:
                    self.logger.warning(
                        f"Table {idx} extraction error on page {page.number}: {table_content['error']}"
                    )
                else:
                    self.logger.info(
                        f"Successfully extracted table {idx} from page {page.number}:\n"
                        f"Dimensions: {table_content['num_rows']}x{table_content['num_cols']} cells\n"
                        f"Content preview: {str(table_content['structured_content'])[:200]}..."
                    )
                tables.append({
                    "id": f"table_{page.number}_{idx}",
                    "page": page.number,
                    "position": {
                        "x0": boundary.x0,
                        "y0": boundary.y0,
                        "x1": boundary.x1,
                        "y1": boundary.y1
                    },
                    "content": table_content,
                    "confidence": boundary.confidence
                })

        self.logger.info(
            f"Successfully processed {len(tables)} tables on page {page.number}"
        )
        return tables

    def _detect_tables(self, img: np.ndarray) -> List[TableBoundary]:
        """Detect table boundaries in the image."""
        self.logger.debug(f"Detecting tables in image of size {img.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect lines
        horizontal = self._detect_lines(binary, "horizontal")
        vertical = self._detect_lines(binary, "vertical")
        
        # Combine lines
        table_mask = cv2.add(horizontal, vertical)
        
        # Find contours
        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process contours to find table boundaries
        boundaries = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            # Extract region for confidence calculation
            region = table_mask[y:y+h, x:x+w]
            h_region = horizontal[y:y+h, x:x+w]
            v_region = vertical[y:y+h, x:x+w]
            
            confidence = self._calculate_confidence(region, h_region, v_region)
            
            # Only include tables above minimum confidence threshold
            if confidence >= self.min_confidence:
                boundaries.append(TableBoundary(
                    x0=float(x),
                    y0=float(y),
                    x1=float(x + w),
                    y1=float(y + h),
                    confidence=confidence
                ))
        
        return boundaries

    def _detect_lines(self, img: np.ndarray, direction: str) -> np.ndarray:
        """Detect horizontal or vertical lines in the image."""
        kernel_length = img.shape[1]//40 if direction == "horizontal" else img.shape[0]//40
        
        # Create kernel
        kernel = np.ones((1, kernel_length)) if direction == "horizontal" \
                else np.ones((kernel_length, 1))
        
        # Apply morphology operations
        morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        return morphed

    def _calculate_confidence(self, 
                            region: np.ndarray, 
                            h_lines: np.ndarray, 
                            v_lines: np.ndarray) -> float:
        """Calculate confidence score for table detection."""
        # Count line intersections
        intersections = cv2.bitwise_and(h_lines, v_lines)
        intersection_points = np.sum(intersections > 0)
        
        # Count lines
        h_line_count = np.sum(h_lines > 0) / h_lines.shape[1]
        v_line_count = np.sum(v_lines > 0) / v_lines.shape[0]
        
        # Calculate grid regularity
        if h_line_count > 0 and v_line_count > 0:
            grid_score = intersection_points / (h_line_count * v_line_count)
        else:
            grid_score = 0
        
        return float(grid_score)

    def _extract_table_content(self, 
                             page: fitz.Page, 
                             boundary: TableBoundary) -> Dict[str, Any]:
        """Extract and structure table content."""
        try:
            # Extract text within the boundary
            table_rect = fitz.Rect(
                boundary.x0,
                boundary.y0,
                boundary.x1,
                boundary.y1
            )
            words = page.get_text("words", clip=table_rect)
            
            # Group words into cells
            cells = self._group_words_into_cells(words, boundary)
            
            # Convert cells to structured format
            df = self._cells_to_dataframe(cells)
            
            return {
                "structured_content": df.to_dict('records'),
                "raw_text": page.get_text("text", clip=table_rect),
                "num_rows": len(df),
                "num_cols": len(df.columns)
            }
        
        except Exception as e:
            return {
                "error": f"Failed to extract table content: {str(e)}",
                "raw_text": page.get_text("text", clip=table_rect)
            }

    def _group_words_into_cells(self, 
                               words: List, 
                               boundary: TableBoundary) -> List[Dict[str, Any]]:
        """Group words into table cells based on their position."""
        cells = []
        current_row = []
        last_y = None
        row_height_threshold = 5  # Adjust based on your needs
        
        # Sort words by y-coordinate first, then x-coordinate
        sorted_words = sorted(words, key=lambda w: (w[3], w[0]))
        
        for word in sorted_words:
            x0, y0, x1, y1 = word[0:4]
            text = word[4]
            
            # Check if we're starting a new row
            if last_y is None or abs(y0 - last_y) > row_height_threshold:
                if current_row:
                    cells.append(sorted(current_row, key=lambda c: c['x']))
                current_row = []
                last_y = y0
            
            current_row.append({
                'x': x0,
                'y': y0,
                'text': text
            })
        
        # Don't forget to append the last row
        if current_row:
            cells.append(sorted(current_row, key=lambda c: c['x']))
        
        return cells

    def _cells_to_dataframe(self, cells: List[List[Dict]]) -> pd.DataFrame:
        """Convert cell groups to a pandas DataFrame."""
        # Create matrix representation
        matrix = []
        for row in cells:
            matrix.append([cell['text'] for cell in row])
        
        # Convert to DataFrame
        df = pd.DataFrame(matrix)
        
        # Try to use first row as header if it appears to be one
        if len(df) > 1:
            if self._is_header_row(df.iloc[0]):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
        
        return df

    def _is_header_row(self, row: pd.Series) -> bool:
        """Determine if a row is likely a header row."""
        # Simple heuristic: check if the row has different formatting
        # or if its values are unique and non-numeric
        try:
            row.astype(float)
            return False
        except:
            return len(set(row)) == len(row)

    async def detect_and_analyze_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect tables and analyze their structure using Table Transformer."""
        # Prepare image for the model
        image_tensor = self.transform_image(image).unsqueeze(0).to(self.device)
        
        # Detect tables
        with torch.no_grad():
            table_outputs = self.table_detector(image_tensor)
            structure_outputs = self.structure_recognizer(image_tensor)
        
        # Process detections
        tables = []
        for table_box, structure in zip(
            table_outputs['pred_boxes'], 
            structure_outputs['pred_structures']
        ):
            table_info = {
                "bbox": table_box.tolist(),
                "cells": self._analyze_table_structure(structure),
                "confidence": float(table_outputs['pred_scores'].max())
            }
            tables.append(table_info)
            
        return tables

    def _extract_cell_contents(self, image: Image.Image, cells: List[Dict]) -> pd.DataFrame:
        """Extract text from cells using PaddleOCR and structure into DataFrame."""
        cell_contents = []
        
        for cell in cells:
            # Crop cell region
            cell_image = image.crop((
                cell['bbox'][0], cell['bbox'][1],
                cell['bbox'][2], cell['bbox'][3]
            ))
            
            # Extract text using PaddleOCR
            ocr_result = self.ocr_engine.ocr(
                np.array(cell_image), 
                cls=True
            )
            
            cell_text = ' '.join([line[1][0] for line in ocr_result])
            cell_contents.append({
                'row': cell['row_index'],
                'col': cell['col_index'],
                'text': cell_text,
                'confidence': float(np.mean([line[1][1] for line in ocr_result]))
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(cell_contents)
        pivot_df = df.pivot(index='row', columns='col', values='text')
        return pivot_df.reset_index(drop=True)

    def _analyze_table_structure(self, structure_output: Dict) -> List[Dict]:
        """Analyze table structure from model output."""
        cells = []
        rows = structure_output['rows']
        cols = structure_output['columns']
        
        # Create cell grid
        for i, row in enumerate(rows[:-1]):
            for j, col in enumerate(cols[:-1]):
                cell = {
                    'row_index': i,
                    'col_index': j,
                    'bbox': [
                        cols[j],
                        rows[i],
                        cols[j+1],
                        rows[i+1]
                    ],
                    'is_header': i == 0,  # Assume first row is header
                    'spanning': self._detect_cell_spanning(
                        [cols[j], rows[i], cols[j+1], rows[i+1]],
                        structure_output['spans']
                    )
                }
                cells.append(cell)
        
        return cells

    def _post_process_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to improve table quality."""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Merge similar headers
        df.columns = self._merge_similar_headers(df.columns)
        
        # Fix data types
        df = self._infer_and_convert_dtypes(df)
        
        # Handle merged cells
        df = self._handle_merged_cells(df)
        
        return df

