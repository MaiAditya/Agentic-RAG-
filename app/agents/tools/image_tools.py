from app.core.image_processor import EnhancedImageProcessor

class ImageAnalysisTool(PDFTool):
    name = "image_analysis"
    description = "Analyze and extract information from images in the document"
    
    def __init__(self):
        super().__init__()
        self.processor = EnhancedImageProcessor()
    
    async def _arun(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Process image
            image = Image.open(io.BytesIO(image_data['content']))
            results = await self.processor.process_image(image)
            
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