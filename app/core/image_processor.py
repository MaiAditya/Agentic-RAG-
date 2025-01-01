from typing import List, Dict, Any, Optional, Union
import fitz
import cv2
import numpy as np
from PIL import Image
import io
import base64
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from transformers import DetrImageProcessor, DetrForObjectDetection

@dataclass
class ImageMetadata:
    width: int
    height: int
    format: str
    color_space: str
    bits_per_component: int

@dataclass
class ImageRegion:
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str
    content: Optional[str] = None

class ImageProcessor:
    def __init__(self):
        # Initialize vision models
        try:
            # BLIP-2 for image captioning and understanding
            self.caption_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.caption_model = AutoModelForVision2Seq.from_pretrained(
                "Salesforce/blip2-opt-2.7b", 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # DETR for object detection
            self.object_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.object_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            # Move models to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.caption_model.to(self.device)
            self.object_model.to(self.device)
            
            logger.info(f"Image processing models loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing image processing models: {e}")
            raise

    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image with enhanced capabilities"""
        try:
            # Validate and preprocess image
            if not self._validate_image(image):
                return {"error": "Invalid image quality or format"}

            # Generate detailed caption
            caption = await self._generate_caption(image)
            
            # Detect objects and regions of interest
            regions = await self._detect_objects(image)
            
            # Analyze image quality and characteristics
            quality_metrics = self._analyze_quality(image)
            
            return {
                "caption": caption,
                "regions": [r.__dict__ for r in regions],
                "quality_metrics": quality_metrics,
                "size": image.size,
                "format": image.format,
                "mode": image.mode
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}

    def _validate_image(self, image: Image.Image) -> bool:
        """Enhanced image validation"""
        try:
            # Check minimum dimensions
            if image.size[0] < 100 or image.size[1] < 100:
                logger.warning("Image too small")
                return False
                
            # Check for extremely large images
            if image.size[0] > 4000 or image.size[1] > 4000:
                logger.warning("Image too large, will be resized")
                image.thumbnail((4000, 4000))
                
            # Check image mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"Unsupported image mode: {image.mode}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False

    async def _generate_caption(self, image: Image.Image) -> str:
        """Generate detailed image caption using BLIP-2"""
        try:
            # Prepare image for the model
            inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=50,
                    num_beams=5,
                    length_penalty=1.0
                )
            
            caption = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return "Error generating caption"

    async def _detect_objects(self, image: Image.Image) -> List[ImageRegion]:
        """Detect objects and regions in the image using DETR"""
        try:
            # Prepare image for object detection
            inputs = self.object_processor(images=image, return_tensors="pt").to(self.device)
            
            # Perform object detection
            with torch.no_grad():
                outputs = self.object_model(**inputs)
            
            # Process results
            processed_outputs = self.object_processor.post_process_object_detection(
                outputs,
                threshold=0.7,
                target_sizes=[(image.size[1], image.size[0])]
            )[0]
            
            regions = []
            for score, label, box in zip(
                processed_outputs["scores"],
                processed_outputs["labels"],
                processed_outputs["boxes"]
            ):
                regions.append(ImageRegion(
                    bbox=box.tolist(),
                    score=score.item(),
                    label=self.object_model.config.id2label[label.item()]
                ))
            
            return regions
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []

    def _analyze_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image quality metrics"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            return {
                "brightness": float(np.mean(img_array)),
                "contrast": float(np.std(img_array)),
                "sharpness": self._calculate_sharpness(img_array),
                "aspect_ratio": image.size[0] / image.size[1],
                "resolution": image.size[0] * image.size[1]
            }
            
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return {}

    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)
            
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            return float(np.var(np.abs(self._convolve2d(img_array, laplacian))))
            
        except Exception as e:
            logger.error(f"Sharpness calculation error: {e}")
            return 0.0

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution implementation"""
        # Implementation details...
        return np.zeros_like(image)  # Placeholder

    async def process(self, image_data: Union[bytes, 'fitz.Page']) -> Optional[List[Dict[str, Any]]]:
        """Process images from a PDF page or raw bytes"""
        try:
            # Handle PyMuPDF Page object
            if isinstance(image_data, fitz.Page):
                # Get list of image blocks from the page
                image_list = image_data.get_images()
                
                if not image_list:
                    logger.info("No images found on page")
                    return None
                    
                processed_images = []
                for img_idx, image_info in enumerate(image_list):
                    try:
                        # Get the image data
                        xref = image_info[0]
                        base_image = image_data.parent.extract_image(xref)
                        
                        if base_image:
                            image_bytes = base_image["image"]
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Process the image using our existing method
                            results = await self.process_image(image)
                            
                            if "error" not in results:
                                processed_images.append({
                                    "caption": results["caption"],
                                    "metadata": {
                                        "size": results["size"],
                                        "format": results["format"],
                                        "mode": results["mode"],
                                        "quality": results["quality_metrics"],
                                        "regions": results["regions"]
                                    }
                                })
                                
                    except Exception as e:
                        logger.error(f"Error processing image {img_idx} on page: {e}")
                        continue
                        
                return processed_images if processed_images else None
                
            else:
                # Handle direct bytes input
                image = Image.open(io.BytesIO(image_data))
                results = await self.process_image(image)
                
                if "error" in results:
                    return None
                    
                return [{
                    "caption": results["caption"],
                    "metadata": {
                        "size": results["size"],
                        "format": results["format"],
                        "mode": results["mode"],
                        "quality": results["quality_metrics"],
                        "regions": results["regions"]
                    }
                }]
                
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            return None
