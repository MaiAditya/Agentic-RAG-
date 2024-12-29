from langchain.tools import BaseTool
from typing import Dict, Any
from abc import abstractmethod

class PDFTool(BaseTool):
    @abstractmethod
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Run the tool asynchronously"""
        pass 