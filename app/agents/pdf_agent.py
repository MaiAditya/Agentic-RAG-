from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import logging
import os

# Set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an AI assistant helping with questions about a PDF document.

Context from the document:
{context}

Human Question: {query}

Please provide a clear and concise answer based on the context above. If the context doesn't contain relevant information to answer the question, please say so.

Answer:"""

class PDFAgent:
    def __init__(self, pdf_processor, chroma_client):
        self.pdf_processor = pdf_processor
        self.chroma_client = chroma_client
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    async def process_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process a PDF file and store its contents."""
        try:
            extracted_content = await self.pdf_processor.process(pdf_content)
            await self.chroma_client.add_documents([extracted_content])
            return {
                "status": "success",
                "message": "PDF processed successfully"
            }
        except Exception as e:
            logger.error(f"Error in process_pdf: {e}")
            raise

    async def answer_query(self, query: str) -> str:
        """Answer a query using RAG."""
        try:
            # Query the vector database with a smaller limit
            db_results = await self.chroma_client.query(query, limit=3)
            
            # Format the context
            context = self._format_context(db_results)
            
            if not context:
                return "I couldn't find any relevant information in the document to answer your question."

            # Create the prompt with context and query
            chain = self.prompt | self.llm
            
            # Get the response
            response = await chain.ainvoke({
                "context": context,
                "query": query
            })
            
            return response.content

        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            raise

    def _format_context(self, results: Dict[str, Any]) -> str:
        """Format the search results into a clear context string."""
        context_parts = []

        # Process text content
        if "text" in results and results["text"]["documents"]:
            text_parts = []
            for doc, metadata in zip(results["text"]["documents"], results["text"]["metadatas"]):
                if doc.strip():  # Only include non-empty text
                    page_num = metadata.get("page_num", "unknown")
                    text_parts.append(f"[Page {page_num}] {doc.strip()}")
            if text_parts:
                context_parts.append("Text Content:\n" + "\n".join(text_parts))

        # Process table content
        if "tables" in results and results["tables"]["documents"]:
            table_parts = []
            for table, metadata in zip(results["tables"]["documents"], results["tables"]["metadatas"]):
                if table.strip():
                    page_num = metadata.get("page_num", "unknown")
                    table_parts.append(f"[Page {page_num}] Table content: {table.strip()}")
            if table_parts:
                context_parts.append("\nTable Content:\n" + "\n".join(table_parts))

        # Process image content
        if "images" in results and results["images"]["documents"]:
            image_parts = []
            for img, metadata in zip(results["images"]["documents"], results["images"]["metadatas"]):
                page_num = metadata.get("page_num", "unknown")
                image_parts.append(f"[Page {page_num}] {img}")
            if image_parts:
                context_parts.append("\nImage References:\n" + "\n".join(image_parts))

        return "\n\n".join(context_parts)