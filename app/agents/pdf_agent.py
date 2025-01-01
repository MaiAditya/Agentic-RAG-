from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .tools.document_tools import DocumentSearchTool, TableSearchTool, ImageSearchTool
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from .prompts import REACT_AGENT_PROMPT
from loguru import logger
import uuid

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
            TableSearchTool(chroma_client),
            ImageSearchTool(chroma_client)
        ]
        
        # Initialize prompts
        self.react_prompt = ChatPromptTemplate.from_template(REACT_AGENT_PROMPT)

    async def process_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """Process a PDF file and store its contents."""
        try:
            # Extract content using the PDF processor
            extracted_content = await self.pdf_processor.process(pdf_content)
            if not extracted_content:
                raise ValueError("Failed to extract content from PDF")
            
            # Process each page into separate documents
            documents = []
            for page_num, page in enumerate(extracted_content.get("pages", [])):
                if page.get("text"):
                    text_doc = {
                        "id": f"text_{uuid.uuid4()}",
                        "type": "text",
                        "content": page["text"],
                        "metadata": {
                            "page": page_num,
                            "source": "text",
                            **extracted_content.get("metadata", {})
                        }
                    }
                    documents.append(text_doc)
                
                # Process images
                for img in page.get("images", []):
                    if img.get("caption"):
                        img_doc = {
                            "id": f"image_{uuid.uuid4()}",
                            "type": "images",
                            "content": img["caption"],
                            "metadata": {
                                "page": page_num,
                                "source": "image",
                                **img.get("metadata", {})
                            }
                        }
                        documents.append(img_doc)
                
                # Process tables
                for table in page.get("tables", []):
                    if table.get("content"):
                        table_doc = {
                            "id": f"table_{uuid.uuid4()}",
                            "type": "tables",
                            "content": table["content"],
                            "metadata": {
                                "page": page_num,
                                "source": "table",
                                **table.get("metadata", {})
                            }
                        }
                        documents.append(table_doc)
            
            logger.info(f"Processed {len(documents)} documents from PDF")
            
            # Store in vector database
            if documents:
                await self.chroma_client.add_documents(documents)
            
            return {
                "status": "success",
                "message": "PDF processed successfully",
                "document_count": len(documents),
                "metadata": extracted_content.get("metadata", {}),
                "file_size": len(pdf_content)
            }
        except Exception as e:
            logger.error(f"Error in process_pdf: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")

    async def answer_query(self, query: str) -> str:
        """Answer a query about the processed PDF content.
        
        Args:
            query: The question to answer
            
        Returns:
            str: The generated answer
        """
        try:
            # Search for relevant documents
            search_results = await self.document_search(query)
            
            # Extract relevant documents
            relevant_docs = []
            for collection_results in search_results.values():
                for i, (doc, metadata, distance) in enumerate(zip(
                    collection_results["documents"],
                    collection_results["metadatas"],
                    collection_results["distances"]
                )):
                    relevant_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance
                    })
            
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            # Generate answer using the relevant documents
            answer = await self._generate_answer(query, relevant_docs)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            raise ValueError(f"Failed to answer query: {str(e)}")

    def _format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into a context string."""
        context_parts = []
        for result in results:
            content = result.get("content", "")
            source_type = result.get("type", "text")
            context_parts.append(f"[{source_type.upper()}]: {content}")
        
        return "\n\n".join(context_parts)

    def _format_tools(self) -> str:
        """Format tools description for the prompt."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)

    async def _generate_answer(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the query and relevant documents."""
        try:
            # Prepare context from relevant documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.get('content', '')}" 
                for i, doc in enumerate(relevant_docs)
            ])
            
            # Create messages in the correct format for ChatOpenAI
            messages = [
                {
                    "content": "You are a helpful assistant that answers questions based on the provided documents.",
                    "role": "system"
                },
                {
                    "content": f"""Based on the following documents, please answer this question: {query}

Context from documents:
{context}

Please provide a clear and concise answer based only on the information provided in the documents above.
If the information is not available in the documents, please say so.""",
                    "role": "user"
                }
            ]

            # Get response from LLM using the correct format
            response = await self.llm.ainvoke(
                input=messages
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise ValueError(f"Failed to generate answer: {str(e)}")

    async def document_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant documents across all collections.
        
        Args:
            query: The search query
            top_k: Number of results to return per collection
            
        Returns:
            Dict containing search results for each collection
        """
        try:
            results = {}
            
            # Search each collection
            for collection_name in ["text", "tables", "images"]:
                try:
                    collection_results = await self.chroma_client.query_collection(
                        collection_name=collection_name,
                        query_text=query,
                        n_results=top_k
                    )
                    if collection_results:
                        results[collection_name] = collection_results
                        logger.info(f"Found {len(collection_results.get('documents', []))} results in {collection_name}")
                    
                except Exception as e:
                    logger.warning(f"Error searching {collection_name} collection: {str(e)}")
                    continue
            
            if not results:
                logger.warning("No results found in any collection")
                return {}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            raise ValueError(f"Failed to search documents: {str(e)}")