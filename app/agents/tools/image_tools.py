from PIL import Image
import io
from typing import Dict, Any
from loguru import logger
from app.core.image_processor import ImageProcessor
from app.agents.tools.base import PDFTool
from pydantic import Field

class ImageAnalysisTool(PDFTool):
    name = "image_analysis"
    description = "Analyze and extract detailed information from images including captions, objects, and quality metrics"
    processor: ImageProcessor = Field(default_factory=ImageProcessor)
    
    def __init__(self):
        super().__init__()
    
    def _run(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous implementation - calls async version"""
        raise NotImplementedError("Use async version of this tool")
    
    async def _arun(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze an image using the ImageProcessor"""
        try:
            # Process image
            image = Image.open(io.BytesIO(image_data['content']))
            results = await self.processor.process_image(image)
            
            if "error" in results:
                return {
                    "type": "error",
                    "content": f"Failed to analyze image: {results['error']}"
                }
            
            return {
                "type": "image_analysis",
                "content": {
                    "caption": results.get("caption"),
                    "objects": results.get("regions", []),
                    "quality": results.get("quality_metrics", {}),
                    "metadata": {
                        "size": results.get("size"),
                        "format": results.get("format"),
                        "mode": results.get("mode")
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {
                "type": "error",
                "content": f"Failed to analyze image: {str(e)}"
            } 