from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .tools.document_tools import DocumentSearchTool, TableSearchTool, ImageSearchTool
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from .prompts import REACT_AGENT_PROMPT
from loguru import logger

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
            logger.info(f"Processing query: {query}")
            
            # Get relevant documents using hybrid retrieval
            logger.debug("Fetching relevant documents...")
            results = await self.retriever.get_relevant_documents(query)
            logger.debug(f"Retrieved {len(results)} initial results")
            
            # Rerank results
            logger.debug("Reranking results...")
            reranked_results = self.reranker.rerank(results, query)
            logger.debug(f"Reranked results count: {len(reranked_results)}")

            # Format context from top results
            context = self._format_context(reranked_results[:3])
            logger.debug(f"Formatted context (first 200 chars): {context[:200]}...")
            
            if not context:
                logger.warning("No relevant context found for the query")
                return "I couldn't find relevant information to answer your question."

            # Execute ReAct chain
            logger.debug("Executing ReAct chain...")
            chain = self.react_prompt | self.llm
            response = await chain.ainvoke({
                "tools": self._format_tools(),
                "input": query,
                "context": context,
                "chat_history": ""
            })
            logger.debug(f"LLM Response: {response.content}")
            
            # Parse and execute tool calls from the response
            content = response.content
            if "Action:" in content:
                logger.info(f"Initial LLM Response with Action: {content}")
                # Split only on the first occurrence of "Action Input:"
                action_parts = content.split("Action:")[1].split("Action Input:", 1)[0].strip()
                action_input = content.split("Action Input:", 1)[1].split("Observation:", 1)[0].strip()
                logger.info(f"Parsed Action: {action_parts}")
                logger.info(f"Parsed Action Input: {action_input}")
                
                # Extract just the tool name from the action
                clean_action = action_parts.lower()
                # Remove common prefixes
                for prefix in ["use the ", "use ", "call the ", "call "]:
                    if clean_action.startswith(prefix):
                        clean_action = clean_action[len(prefix):]
                # Remove everything after the first space or "tool"
                clean_action = clean_action.split(" ")[0].replace("tool", "").strip()
                
                logger.info(f"Cleaned action name for matching: {clean_action}")
                
                # Find and execute the requested tool
                tool_observation = None
                for tool in self.tools:
                    logger.debug(f"Comparing tool {tool.name.lower()} with action {clean_action}")
                    if tool.name.lower() == clean_action:
                        logger.info(f"Executing tool: {tool.name}")
                        try:
                            tool_observation = await tool.arun(action_input)
                            logger.info(f"Tool observation result: {tool_observation}")
                        except Exception as e:
                            logger.error(f"Error executing tool: {e}")
                        break
                
                if tool_observation:
                    # Feed the observation back to the LLM for final answer
                    chat_history = f"{content}\nObservation: {tool_observation}"
                    logger.info(f"Feeding back to LLM with chat history: {chat_history}")
                    new_response = await chain.ainvoke({
                        "tools": self._format_tools(),
                        "input": query,
                        "context": context,
                        "chat_history": chat_history
                    })
                    content = new_response.content
                    logger.info(f"New LLM response after tool execution: {content}")
                else:
                    logger.warning(f"No matching tool found for action: {clean_action}")
                    logger.warning(f"Available tools: {[tool.name.lower() for tool in self.tools]}")
            
            # Extract final answer
            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:")[-1].strip()
                logger.info(f"Extracted final answer: {final_answer}")
                return final_answer

            logger.debug("Returning raw content as no final answer found")
            return content

        except Exception as e:
            logger.error(f"Error in answer_query: {e}", exc_info=True)
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
        """Format tools description for the prompt."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)