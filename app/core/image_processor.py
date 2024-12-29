from typing import List, Dict, Any, Optional
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

@dataclass
class ImageMetadata:
    width: int
    height: int
    format: str
    color_space: str
    bits_per_component: int

class ImageProcessor:
    def __init__(self):
        self.min_size = 50  # Reduced from 100 to 50 pixels
        self.quality_threshold = 0.3  # Reduced from 0.5 to 0.3
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
        self.max_workers = 4  # Adjust based on your CPU cores
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    async def process_document(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Process all pages of a document in parallel."""
        all_images = []
        tasks = []
        
        # Create processing tasks for each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            task = asyncio.create_task(self.process(page))
            tasks.append(task)
            
            # Process in batches to manage memory
            if len(tasks) >= self.max_workers:
                batch_results = await asyncio.gather(*tasks)
                all_images.extend([img for page_imgs in batch_results for img in page_imgs])
                tasks = []
        
        # Process remaining tasks
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            all_images.extend([img for page_imgs in batch_results for img in page_imgs])
        
        return all_images

    async def process(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Process a PDF page to extract and analyze images."""
        images = []
        
        try:
            image_list = page.get_images(full=True)
            
            if not image_list:
                logger.debug(f"No images found on page {page.number}")
                return images
            
            # Process images in parallel using ThreadPoolExecutor
            process_func = partial(self._process_image, page=page)
            loop = asyncio.get_event_loop()
            
            # Process images in parallel
            futures = []
            for img_idx, img in enumerate(image_list):
                future = loop.run_in_executor(self.executor, process_func, img_idx, img)
                futures.append(future)
            
            # Gather results
            results = await asyncio.gather(*futures)
            images = [img for img in results if img is not None]
            
            if images:
                logger.info(f"Processed {len(images)} images on page {page.number}")
            
            return images
            
        except Exception as e:
            logger.error(f"Error in image processing on page {page.number}: {e}")
            return []

    def _process_image(self, img_idx: int, img: tuple, page: fitz.Page) -> Optional[Dict[str, Any]]:
        """Process a single image with all necessary steps."""
        try:
            xref = img[0]
            if not xref:
                return None
                
            base_image = page.parent.extract_image(xref)
            if not base_image:
                return None
            
            # Add size information
            base_image["width"] = img[2]
            base_image["height"] = img[3]
            
            if not self._check_image_quality(base_image):
                return None
            
            image_rect = page.get_image_bbox(img)
            processed_image = self._process_single_image(base_image, image_rect)
            
            if processed_image:
                processed_image["id"] = f"image_{page.number}_{img_idx}"
                return processed_image
                
        except Exception as e:
            logger.warning(f"Error processing image {img_idx} on page {page.number}: {e}")
            return None

    def _check_image_quality(self, image_data: Dict) -> bool:
        """Check if the image meets quality requirements."""
        try:
            # Basic validation
            if not isinstance(image_data, dict) or "image" not in image_data:
                return False
            
            # Get dimensions from the image data
            width = image_data.get("width", 0)
            height = image_data.get("height", 0)
            
            # Basic dimension check
            if width <= 0 or height <= 0:
                return False
            
            # Very relaxed size check - if either dimension is reasonable
            if max(width, height) >= self.min_size:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in image quality check: {e}")
            return False

    def _process_single_image(self, base_image: Dict, image_rect: fitz.Rect) -> Optional[Dict[str, Any]]:
        """Optimized version of single image processing."""
        try:
            image_bytes = base_image.get("image")
            if not image_bytes:
                return None
            
            # Process image data
            image = self._prepare_image(image_bytes, base_image.get("ext", ""))
            if not image:
                return None
            
            # Generate all required data in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    'features': executor.submit(self._extract_image_features, image),
                    'description': executor.submit(self._generate_image_description, image),
                    'base64': executor.submit(self._convert_to_base64, image)
                }
                
                results = {k: v.result() for k, v in futures.items()}
            
            return {
                "position": {
                    "x0": image_rect.x0,
                    "y0": image_rect.y0,
                    "x1": image_rect.x1,
                    "y1": image_rect.y1
                },
                "metadata": {
                    "width": image.width,
                    "height": image.height,
                    "format": image.format or "JPEG",
                    "color_space": "RGB",
                    "bits_per_component": 8
                },
                "description": results['description'],
                "features": results['features'],
                "base64_image": results['base64'],
                "image_type": self._determine_image_type(image)
            }
            
        except Exception as e:
            logger.error(f"Error in _process_single_image: {e}")
            return None

    def _prepare_image(self, image_bytes: bytes, format_hint: str) -> Optional[Image.Image]:
        """Prepare image data for processing."""
        try:
            if format_hint.upper() == "JPX":
                try:
                    pix = fitz.Pixmap(image_bytes)
                    if pix.n >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes()
                except Exception:
                    pass
            
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.debug(f"Error preparing image: {str(e)}")
            return None

    def _convert_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG', quality=85)
        return base64.b64encode(buffered.getvalue()).decode()

    def _generate_image_description(self, image: Image.Image) -> str:
        """Generate a description of the image using the vision model."""
        try:
            if self.processor and self.model:
                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Prepare image for the model
                try:
                    # Modified processor call with pixel_values
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    pixel_values = inputs.pixel_values
                    
                    # Generate description
                    outputs = self.model.generate(
                        pixel_values=pixel_values,
                        max_length=50,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode the output
                    description = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    return description
                except Exception as e:
                    logger.error(f"Error in model inference: {e}")
                    return "Error generating image description"
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
                    features.extend(hist.flatten().tolist())  # Convert numpy array to list
            else:
                hist = cv2.calcHist([resized], [0], None, [8], [0, 256])
                features.extend(hist.flatten().tolist())  # Convert numpy array to list
            
            # Edge features
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            else:
                gray = resized
            edges = cv2.Canny(gray, 100, 200)
            edge_features = cv2.calcHist([edges], [0], None, [8], [0, 256])
            features.extend(edge_features.flatten().tolist())  # Convert numpy array to list
            
            return features
        
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
