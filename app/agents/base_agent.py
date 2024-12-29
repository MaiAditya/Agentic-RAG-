from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel

class BaseAgent(ABC):
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data"""
        pass

    @abstractmethod
    async def query(self, query: str) -> Dict[str, Any]:
        """Handle queries"""
        pass
