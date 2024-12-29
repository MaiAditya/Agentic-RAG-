from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .tools.document_tools import DocumentSearchTool, TableSearchTool
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from .prompts import REACT_AGENT_PROMPT

class PDFAgent:
    def __init__(self, pdf_processor, chroma_client):
        # Initialize components
        self.pdf_processor = pdf_processor
        self.chroma_client = chroma_client
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"
        )
        
        # Initialize retrievers and reranker
        self.retriever = HybridRetriever(chroma_client)
        self.reranker = Reranker()
        
        # Initialize tools
        self.tools = [
            DocumentSearchTool(chroma_client),
            TableSearchTool(chroma_client)
        ]
        
        # Initialize prompts
        self.react_prompt = ChatPromptTemplate.from_template(REACT_AGENT_PROMPT)

    async def process_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process a PDF file and store its contents."""
        try:
            # Extract content using the PDF processor
            extracted_content = await self.pdf_processor.process(pdf_content)
            
            # Store in vector database
            await self.chroma_client.add_documents([extracted_content])
            
            return {
                "status": "success",
                "message": "PDF processed successfully",
                "metadata": extracted_content.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error in process_pdf: {e}")
            raise

    async def answer_query(self, query: str) -> str:
        """Answer a query using RAG with the ReAct pattern."""
        try:
            # Get relevant documents using hybrid retrieval
            results = await self.retriever.get_relevant_documents(query)
            
            # Rerank results
            reranked_results = self.reranker.rerank(results, query)
            
            # Format context from top results
            context = self._format_context(reranked_results[:3])
            
            if not context:
                return "I couldn't find relevant information to answer your question."

            # Execute ReAct chain
            chain = self.react_prompt | self.llm
            response = await chain.ainvoke({
                "tools": self._format_tools(),
                "input": query,
                "context": context,
                "chat_history": ""
            })
            
            return response.content

        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            raise

    def _format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into a context string."""
        context_parts = []
        for result in results:
            content = result.get("content", "")
            source_type = result.get("type", "text")
            context_parts.append(f"[{source_type.upper()}]: {content}")
        
        return "\n\n".join(context_parts)

    def _format_tools(self) -> str:
        """Format available tools for the prompt."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)