from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image
import fitz
from dataclasses import dataclass
import pandas as pd
from loguru import logger 
from app.core.models.table_detection import DeepTableDetector, TableRegion

@dataclass
class TableBoundary:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

class TableProcessor:
    def __init__(self):
        self.dl_detector = DeepTableDetector()
        self.min_confidence = 0.5 # Lowered from 0.7 to match detector threshold
        logger.info("Initialized TableProcessor with deep learning models")

    async def process(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Process a PDF page to extract tables using deep learning."""
        try:
            # Convert page to image
            pix = page.get_pixmap(dpi=300)  # Increased DPI for better detection
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Log page processing
            logger.debug(f"Processing page {page.number} for table detection. "
                        f"Image size: {img.size}, DPI: 300")
            
            # Detect tables
            tables = await self.dl_detector.detect_tables(img)
            logger.info(f"Found {len(tables)} potential tables on page {page.number}")
            
            # Process each table
            processed_tables = []
            for idx, table in enumerate(tables):
                if table.confidence < self.min_confidence:
                    logger.debug(f"Skipping table {idx} due to low confidence: {table.confidence:.3f}")
                    continue
                    
                logger.info(
                    f"Processing table {idx + 1}/{len(tables)} on page {page.number}:\n"
                    f"Position: ({table.bbox[0]:.1f}, {table.bbox[1]:.1f}, "
                    f"{table.bbox[2]:.1f}, {table.bbox[3]:.1f})\n"
                    f"Confidence: {table.confidence:.2f}"
                )
                
                # Analyze table structure
                structure = await self.dl_detector.analyze_structure(img, table)
                
                if structure:
                    logger.debug(f"Table {idx} structure analysis successful: "
                               f"{structure['num_rows']}x{structure['num_cols']} cells")
                    processed_tables.append({
                        "id": f"table_{page.number}_{idx}",
                        "page": page.number,
                        "position": {
                            "x0": table.bbox[0],
                            "y0": table.bbox[1],
                            "x1": table.bbox[2],
                            "y1": table.bbox[3]
                        },
                        "content": structure,
                        "confidence": table.confidence
                    })
                else:
                    logger.warning(f"Failed to analyze structure for table {idx} on page {page.number}")
                    
            return processed_tables
            
        except Exception as e:
            logger.error(f"Error processing page {page.number}: {str(e)}")
            # Fallback to traditional method if deep learning fails
            logger.info("Attempting fallback to traditional table detection method")
            return await self._fallback_process(page)

    async def _fallback_process(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Traditional table detection method as fallback."""
        try:
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Detect tables using traditional method
            table_boundaries = self._detect_tables(img_np)
            logger.info(f"Fallback method found {len(table_boundaries)} tables on page {page.number}")
            
            # Process detected tables
            processed_tables = []
            for idx, boundary in enumerate(table_boundaries):
                logger.debug(f"Processing fallback table {idx} with confidence {boundary.confidence:.3f}")
                
                if boundary.confidence >= self.min_confidence:
                    table_content = self._extract_table_content(page, boundary)
                    if table_content:
                        processed_tables.append({
                            "id": f"table_{page.number}_{idx}_fallback",
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
            
            return processed_tables
            
        except Exception as e:
            logger.error(f"Fallback processing failed for page {page.number}: {str(e)}")
            return []

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
            area = w * h
            page_area = img.shape[0] * img.shape[1]
            
            self.logger.debug(
                f"Analyzing contour {idx + 1}/{len(contours)}: "
                f"Area ratio: {(area/page_area):.3f}"
            )
            
            # Filter out too small or too large areas
            if 0.01 * page_area < area < 0.9 * page_area:
                confidence = self._calculate_confidence(
                    binary[y:y+h, x:x+w],
                    horizontal[y:y+h, x:x+w],
                    vertical[y:y+h, x:x+w]
                )
                
                self.logger.debug(f"Contour {idx} confidence: {confidence:.2f}")
                
                if confidence > self.min_confidence:
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

