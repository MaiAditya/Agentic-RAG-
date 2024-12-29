from typing import List, Dict, Any
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
@dataclass
class ImageMetadata:
    width: int
    height: int
    format: str
    color_space: str
    bits_per_component: int

class ImageProcessor:
    def __init__(self):
        self.min_size = 100  # Minimum size in pixels for image extraction
        self.quality_threshold = 0.5
        # Initialize the vision model for image captioning
        try:
            self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self.model = AutoModelForVision2Seq.from_pretrained("microsoft/git-base")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            self.processor = None
            self.model = None

    async def process(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Process a PDF page to extract and analyze images."""
        images = []
        
        try:
            # Get list of images on the page
            image_list = page.get_images(full=True)
            
            if not image_list:
                logger.info("No images found on page")
                return images
            
            for img_idx, img in enumerate(image_list):
                try:
                    if not img or not img[0]:
                        logger.warning(f"Invalid image data at index {img_idx}")
                        continue
                        
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    
                    if not base_image:
                        logger.warning(f"Failed to extract image at index {img_idx}")
                        continue
                    
                    if self._check_image_quality(base_image):
                        image_rect = page.get_image_bbox(img)
                        processed_image = self._process_single_image(base_image, image_rect)
                        
                        if processed_image:
                            processed_image["id"] = img_idx
                            images.append(processed_image)
                    else:
                        logger.info(f"Image {img_idx} did not meet quality requirements")
                    
                except Exception as e:
                    logger.warning(f"Error processing image {img_idx}: {e}", exc_info=True)
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Error in image processing: {e}", exc_info=True)
            return []

    def _check_image_quality(self, image_data: Dict) -> bool:
        """Check if the image meets quality requirements."""
        try:
            # Basic size check
            if not image_data.get("width") or not image_data.get("height"):
                return False
            
            if image_data["width"] < self.min_size or image_data["height"] < self.min_size:
                return False
            
            # Check for valid image data
            if not image_data.get("image"):
                return False
            
            # Check color depth
            if image_data.get("bpc", 0) < 4:  # Minimum bits per component
                return False
            
            # Check data size
            min_data_size = 100  # Minimum bytes for a valid image
            if len(image_data["image"]) < min_data_size:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in image quality check: {e}")
            return False

    def _process_single_image(self, 
                            base_image: Dict, 
                            image_rect: fitz.Rect) -> Dict[str, Any]:
        """Process a single image extracted from the PDF."""
        try:
            # Validate image data
            if not base_image.get("image"):
                logger.warning("No image data found in base_image")
                return None
            
            # Get image format and color space
            image_format = base_image.get("ext", "").upper()
            colorspace = base_image.get("colorspace")
            
            try:
                # More robust image conversion
                image_bytes = base_image["image"]
                if image_format == "JPEG":
                    # Direct JPEG handling
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # Default handling
                    mode = "RGB" if colorspace == 3 else "L"
                    image = Image.frombytes(
                        mode,
                        [base_image["width"], base_image["height"]],
                        image_bytes
                    )
                
                # Generate metadata
                metadata = ImageMetadata(
                    width=base_image["width"],
                    height=base_image["height"],
                    format=image_format,
                    color_space=self._get_colorspace_name(colorspace),
                    bits_per_component=base_image.get("bpc", 8)
                )
                
                # Generate image description
                description = self._generate_image_description(image)
                
                # Convert image to base64 for storage/transmission
                buffered = io.BytesIO()
                image.save(buffered, format=image_format if image_format else "PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Extract features for vector search
                features = self._extract_image_features(image)
                
                return {
                    "position": {
                        "x0": image_rect.x0,
                        "y0": image_rect.y0,
                        "x1": image_rect.x1,
                        "y1": image_rect.y1
                    },
                    "metadata": vars(metadata),
                    "description": description,
                    "features": features,
                    "base64_image": img_str,
                    "image_type": self._determine_image_type(image)
                }
                
            except IOError as e:
                logger.error(f"IOError processing image: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error processing single image: {e}")
            return None

    def _generate_image_description(self, image: Image.Image) -> str:
        """Generate a description of the image using the vision model."""
        try:
            if self.processor and self.model:
                # Prepare image for the model
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                # Generate description
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode the output
                description = self.processor.decode(outputs[0], skip_special_tokens=True)
                return description
            else:
                return "Image description not available - vision model not loaded"
        
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return "Error generating image description"

    def _extract_image_features(self, image: Image.Image) -> List[float]:
        """Extract features from the image for vector search."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Resize for consistent feature extraction
            resized = cv2.resize(img_array, (224, 224))
            
            # Extract basic features
            features = []
            
            # Color histogram
            if len(resized.shape) == 3:
                for i in range(3):
                    hist = cv2.calcHist([resized], [i], None, [8], [0, 256])
                    features.extend(hist.flatten())
            else:
                hist = cv2.calcHist([resized], [0], None, [8], [0, 256])
                features.extend(hist.flatten())
            
            # Edge features
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) if len(resized.shape) == 3 else resized
            edges = cv2.Canny(gray, 100, 200)
            edge_features = cv2.calcHist([edges], [0], None, [8], [0, 256])
            features.extend(edge_features.flatten())
            
            return features.tolist()
        
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return []

    def _determine_image_type(self, image: Image.Image) -> str:
        """Determine the type of image (photo, diagram, chart, etc.)."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic heuristics for image type detection
            if len(img_array.shape) == 2:
                return "grayscale"
            
            # Calculate various metrics
            edges = cv2.Canny(img_array, 100, 200)
            edge_density = np.mean(edges > 0)
            color_variance = np.var(img_array)
            
            if edge_density > 0.1 and color_variance < 1000:
                return "diagram"
            elif edge_density > 0.05 and color_variance > 1000:
                return "chart"
            else:
                return "photo"
        
        except Exception as e:
            logger.error(f"Error determining image type: {e}")
            return "unknown"

    @staticmethod
    def _get_colorspace_name(colorspace: int) -> str:
        """Convert colorspace number to name."""
        colorspace_map = {
            1: "GRAY",
            3: "RGB",
            4: "CMYK"
        }
        return colorspace_map.get(colorspace, "unknown")
