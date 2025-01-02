from transformers import (
    DetrImageProcessor, 
    DetrForObjectDetection,
    TableTransformerForObjectDetection
)
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
import datetime
import cv2
import os

@dataclass
class TableRegion:
    bbox: List[float]
    confidence: float
    structure: Dict[str, Any] = None

class DeepTableDetector:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing DeepTableDetector on {self.device}")
        
        # Initialize Table Transformer for table detection
        model_name = "microsoft/table-transformer-detection"
        self.detector_processor = DetrImageProcessor.from_pretrained(
            model_name,
            do_resize=True,
            size={'shortest_edge': 800, 'longest_edge': 1333},
        )
        self.detector_model = TableTransformerForObjectDetection.from_pretrained(model_name).to(self.device)
        self.detection_threshold = 0.5  # Adjusted threshold based on model recommendations
        
        # Initialize Table Structure Recognition
        structure_model = "microsoft/table-transformer-structure-recognition"
        self.structure_processor = DetrImageProcessor.from_pretrained(
            structure_model,
            do_resize=True,
            size={'shortest_edge': 800, 'longest_edge': 1333},
        )
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model).to(self.device)
        self.structure_threshold = 0.5 # Threshold for cell detection
        
        logger.info("Successfully initialized table detection and structure models")
        
    async def detect_tables(self, image: Image.Image) -> List[TableRegion]:
        """Detect tables in the image using Table Transformer."""
        try:
            # Save original image for reference
            os.makedirs("logs/table_detections/originals", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = f"logs/table_detections/originals/original_{timestamp}.png"
            image.save(original_path)
            logger.info(f"Saved original image to {original_path}")
            logger.debug(f"Processing image of size: {image.size}")
            
            # Prepare image and log preprocessing details
            inputs = self.detector_processor(images=image, return_tensors="pt")
            preprocessed_size = inputs['pixel_values'].shape
            logger.debug(f"Preprocessed image tensor shape: {preprocessed_size}")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get and log raw model outputs
            with torch.no_grad():
                outputs = self.detector_model(**inputs)
                logger.debug(f"Raw model output scores shape: {outputs.logits.shape}")
                logger.debug(f"Raw model output boxes shape: {outputs.pred_boxes.shape}")
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detector_processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=self.detection_threshold
            )[0]
            
            # Log detailed detection results with safe score range calculation
            if len(results['scores']) > 0:
                score_min = results['scores'].min().item()
                score_max = results['scores'].max().item()
                score_range = f"{score_min:.3f} to {score_max:.3f}"
            else:
                score_range = "N/A (no detections)"
            
            logger.debug(
                f"Detection results:\n"
                f"- Number of detections: {len(results['scores'])}\n"
                f"- Score range: {score_range}\n"
                f"- Unique labels: {torch.unique(results['labels']).tolist() if len(results['labels']) > 0 else []}"
            )
            
            # Create visualization directory structure
            vis_dir = f"logs/table_detections/visualizations/{timestamp}"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create a copy of the image for visualization
            vis_image = image.copy()
            vis_array = np.array(vis_image)
            
            # Convert to TableRegion objects with enhanced logging
            tables = []
            for idx, (score, label, box) in enumerate(zip(
                results["scores"], 
                results["labels"], 
                results["boxes"]
            )):
                score_val = score.cpu().item()
                label_val = label.cpu().item()
                bbox = box.cpu().tolist()
                
                logger.debug(
                    f"Detection {idx + 1}:\n"
                    f"- Label: {label_val}\n"
                    f"- Score: {score_val:.3f}\n"
                    f"- Bbox: {[f'{x:.1f}' for x in bbox]}"
                )
                
                if label_val == 0:  # Table class
                    tables.append(TableRegion(
                        bbox=bbox,
                        confidence=score_val
                    ))
                    
                    # Draw rectangle and save individual detection
                    x0, y0, x1, y1 = [int(coord) for coord in bbox]
                    cv2.rectangle(
                        vis_array,
                        (x0, y0),
                        (x1, y1),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        vis_array,
                        f"Table {idx}: {score_val:.2f}",
                        (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    
                    # Save cropped table image
                    table_crop = image.crop((x0, y0, x1, y1))
                    crop_path = f"{vis_dir}/table_{idx}_{score_val:.2f}.png"
                    table_crop.save(crop_path)
                    logger.info(f"Saved cropped table {idx} to {crop_path}")
            
            # Save final visualization with all detections
            if tables:
                vis_path = f"{vis_dir}/all_detections.png"
                cv2.imwrite(vis_path, cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR))
                logger.info(
                    f"Detection summary:\n"
                    f"- Found {len(tables)} tables\n"
                    f"- Visualization saved to {vis_path}\n"
                    f"- Individual crops saved in {vis_dir}"
                )
            else:
                logger.warning(
                    "No tables detected. Check:\n"
                    f"- Detection threshold: {self.detection_threshold}\n"
                    f"- Image size: {image.size}\n"
                    f"- Original image saved at: {original_path}"
                )
            
            return tables
            
        except Exception as e:
            logger.error(f"Table detection error: {str(e)}", exc_info=True)
            return []

    async def analyze_structure(
        self, 
        image: Image.Image, 
        region: TableRegion
    ) -> Dict[str, Any]:
        """Analyze table structure using Table Transformer."""
        try:
            # Crop and process table region
            x0, y0, x1, y1 = [int(coord) for coord in region.bbox]
            table_image = image.crop((x0, y0, x1, y1))
            
            # Log table region details
            logger.debug(f"Analyzing table structure for region: {x0},{y0},{x1},{y1}")
            logger.debug(f"Cropped table size: {table_image.size}")
            
            # Prepare image for structure recognition
            inputs = self.structure_processor(images=table_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get structure predictions
            with torch.no_grad():
                outputs = self.structure_model(**inputs)
            
            # Process structure predictions
            target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
            processed_outputs = self.structure_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.structure_threshold
            )[0]
            
            # Log structure detection results
            logger.debug(f"Found {len(processed_outputs['scores'])} potential cells")
            
            # Extract cells and their positions
            cells = self._process_structure_output(processed_outputs, table_image.size)
            structure = self._create_table_structure(cells)
            
            if structure:
                logger.info(
                    f"Successfully analyzed table structure: "
                    f"{structure['num_rows']}x{structure['num_cols']} cells"
                )
            return structure
            
        except Exception as e:
            logger.error(f"Structure analysis error: {str(e)}")
            return None

    def _process_structure_output(
        self, 
        outputs: Dict[str, torch.Tensor], 
        image_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Process model outputs to get cell information."""
        cells = []
        
        for score, label, box in zip(
            outputs["scores"], 
            outputs["labels"], 
            outputs["boxes"]
        ):
            score_val = score.item()
            if score_val > self.structure_threshold:
                bbox = box.tolist()
                cells.append({
                    "bbox": bbox,
                    "type": "cell",
                    "confidence": score_val
                })
                logger.debug(f"Detected cell with confidence {score_val:.3f} at {bbox}")
        
        logger.debug(f"Processed {len(cells)} valid cells")
        return cells

    def _create_table_structure(
        self, 
        cells: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert cell information to structured table format."""
        if not cells:
            return None
            
        # Sort cells by vertical position first, then horizontal
        sorted_cells = sorted(cells, key=lambda c: (c["bbox"][1], c["bbox"][0]))
        
        # Group cells into rows based on vertical position
        rows = []
        current_row = []
        last_y = None
        row_threshold = 10  # Increased threshold for better row detection
        
        for cell in sorted_cells:
            y_center = (cell["bbox"][1] + cell["bbox"][3]) / 2
            
            if last_y is None or abs(y_center - last_y) > row_threshold:
                if current_row:
                    rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
                current_row = []
                last_y = y_center
            
            current_row.append(cell)
        
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
        
        structure = {
            "num_rows": len(rows),
            "num_cols": max(len(row) for row in rows) if rows else 0,
            "cells": cells,
            "rows": rows
        }
        
        logger.debug(
            f"Created table structure with {structure['num_rows']} rows "
            f"and {structure['num_cols']} columns"
        )
        return structure 