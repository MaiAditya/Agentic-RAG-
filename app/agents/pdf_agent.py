from typing import Dict, Any, List, Optional, Tuple
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .tools.document_tools import DocumentSearchTool, TableSearchTool, ImageSearchTool
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from .prompts import REACT_AGENT_PROMPT
from loguru import logger
import uuid
import asyncio
import json
import re

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

    async def answer_query(
        self,
        query: str,
        min_similarity: float = 0.3,
        content_types: List[str] = None,
        page: Optional[int] = None
    ) -> str:
        """Answer a query using both REACT agent with tools and hybrid retriever in parallel."""
        try:
            if content_types is None:
                content_types = ["text", "tables", "images"]
            
            logger.info(f"""
            Starting query execution:
            Query: {query}
            Content Types: {content_types}
            Min Similarity: {min_similarity}
            Page Filter: {page}
            """)
            
            # Task 1: Get results using REACT agent with tools
            tool_results = await self._execute_react_agent(query, content_types)
            
            # Task 2: Get results using hybrid retriever
            hybrid_results = await self.retriever.get_relevant_documents(
                query=query,
                limit=10
            )
            
            logger.info(f"""
            Retrieved results from both methods:
            REACT Agent Results: {len(tool_results)} documents
            Hybrid Retriever Results: {len(hybrid_results)} documents
            """)
            
            # Combine and deduplicate results
            all_results = self._merge_results(hybrid_results, tool_results)
            
            if not all_results:
                return "I couldn't find any relevant information in the document to answer your question."
            
            # Apply reranking to improve result ordering
            reranked_docs = self.reranker.rerank(all_results, query)
            top_docs = reranked_docs[:7]
            
            # Generate final answer using combined and reranked documents
            answer = await self._generate_answer(query, top_docs)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            raise ValueError(f"Failed to answer query: {str(e)}")

    async def _execute_react_agent(self, query: str, content_types: List[str]) -> List[Dict[str, Any]]:
        """Execute the REACT agent to use tools intelligently."""
        try:
            tools_description = self._format_tools()
            logger.info("Starting REACT agent execution")
            
            all_results = []
            max_iterations = 3
            
            messages = [
                {
                    "role": "system",
                    "content": REACT_AGENT_PROMPT.format(
                        tools=tools_description,
                        input=query,
                        context="",
                        chat_history=""
                    )
                }
            ]
            
            for iteration in range(max_iterations):
                response = await self.llm.ainvoke(messages)
                response_text = response.content
                
                logger.info(f"REACT Agent Iteration {iteration + 1}:\n{response_text}")
                
                thought, action, action_input = self._parse_react_response(response_text)
                
                if action is None:
                    logger.warning("No action found in response")
                    break
                    
                if action.lower() == "final answer":
                    if len(all_results) == 0:
                        # If no results yet, force another iteration of tool use
                        messages.append({
                            "role": "user",
                            "content": "Please use the search tools first to find specific information before providing a final answer."
                        })
                        continue
                    else:
                        logger.info("REACT agent finished reasoning with results")
                        break
                
                # Execute the chosen tool
                tool_result = await self._execute_tool(action, action_input)
                
                if tool_result:
                    logger.info(f"Tool execution successful: {json.dumps(tool_result, indent=2)}")
                    if isinstance(tool_result.get('content', []), list):
                        all_results.extend(tool_result['content'])
                    else:
                        all_results.append(tool_result)
                
                # Add the observation to the conversation
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {json.dumps(tool_result, indent=2) if tool_result else 'No results found'}"
                })
            
            logger.info(f"REACT agent gathered {len(all_results)} results")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in REACT agent execution: {str(e)}")
            return []

    def _parse_react_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse the REACT agent's response to extract thought, action, and action input."""
        try:
            # Initialize default values
            thought = None
            action = None
            action_input = None
            
            # Extract thought
            thought_match = re.search(r"Thought:(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()
                
            # Check for Final Answer
            if "Final Answer:" in response:
                return thought, "Final Answer", None
                
            # Extract action
            action_match = re.search(r"Action:(.+?)(?=Action Input:|$)", response, re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()
                
            # Extract action input
            action_input_match = re.search(r"Action Input:(.+?)(?=Observation:|$)", response, re.DOTALL)
            if action_input_match:
                action_input = action_input_match.group(1).strip()
            
            logger.debug(f"""
            Parsed REACT response:
            Thought: {thought}
            Action: {action}
            Action Input: {action_input}
            """)
            
            return thought, action, action_input
            
        except Exception as e:
            logger.error(f"Error parsing REACT response: {str(e)}")
            return None, None, None

    async def _execute_tool(self, action: str, action_input: str) -> Optional[Dict[str, Any]]:
        """Execute the specified tool with the given input."""
        try:
            # Find the appropriate tool
            tool = next((t for t in self.tools if t.name.lower() == action.lower()), None)
            
            if not tool:
                logger.warning(f"Tool not found: {action}")
                return None
                
            logger.info(f"Executing tool: {tool.name} with input: {action_input}")
            
            # Clean up action input
            cleaned_input = str(action_input).strip('"\'').strip()
            
            # Execute the tool and get results
            result = await tool._arun(cleaned_input)
            
            if result:
                logger.info(f"Tool execution successful: {json.dumps(result, indent=2)}")
                
                # Handle different result formats
                if isinstance(result, dict):
                    content = result.get('content', [])
                    if isinstance(content, list):
                        # If content is already a list, return as is
                        return result
                    else:
                        # If content is not a list, wrap it in a list
                        return {
                            "type": result.get("type", "text"),
                            "content": [content] if content else [],
                            "source": result.get("source", tool.name)
                        }
                else:
                    # If result is not a dict, wrap it in our standard format
                    return {
                        "type": "text",
                        "content": [str(result)],
                        "source": tool.name
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing tool {action}: {str(e)}", exc_info=True)
            return None

    def _format_tools(self) -> str:
        """Format tools description for the prompt."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)

    async def _generate_answer(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the query and relevant documents."""
        try:
            # Sort documents by score
            sorted_docs = sorted(relevant_docs, key=lambda x: x.get('score', 0.0), reverse=True)
            
            # Prepare context with source information
            context_parts = []
            for i, doc in enumerate(sorted_docs[:7]):
                content = doc.get('content', '')
                source_type = doc.get('type', 'text')
                source_method = doc.get('source_method', 'unknown')
                
                context_parts.append(
                    f"Document {i+1} [{source_type.upper()}] (via {source_method}):\n{content}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Create messages for the LLM
            messages = [
                            {
                                "content": """You are a helpful assistant that answers questions based on the provided documents. 
                                When multiple sources provide information, synthesize them for a complete answer.""",
                                "role": "system"
                            },
                            {
                                "content": f"""Based on the following documents, please answer this question: {query}

            Context from documents:
            {context}

            Please provide a clear and concise answer based only on the information provided in the documents above.
            If different sources provide complementary information, combine them in your answer.
            If the information is not available in the documents, please say so.""",
                                "role": "user"
                            }
            ]

            # Get response from LLM
            response = await self.llm.ainvoke(
                input=messages
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise ValueError(f"Failed to generate answer: {str(e)}")

    async def document_search(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.3,
        content_types: List[str] = None,
        page: Optional[int] = None
    ) -> Dict[str, Any]:
        """Search for relevant documents across collections.
        
        Args:
            query: The search query
            top_k: Number of results to return per collection
            min_similarity: Minimum similarity threshold
            content_types: List of content types to search
            page: Optional page number to restrict search
            
        Returns:
            Dict containing search results for each collection
        """
        try:
            results = {}
            
            # Search specified collections
            for collection_name in content_types:
                if collection_name not in ["text", "tables", "images"]:
                    continue
                    
                try:
                    collection_results = await self.chroma_client.query_collection(
                        collection_name=collection_name,
                        query_text=query,
                        n_results=top_k,
                        where={"page": page} if page is not None else None
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


    def _merge_results(self, hybrid_results: List[Dict], tool_results: List[Dict]) -> List[Dict]:
        """Merge and deduplicate results from both methods."""
        try:
            merged = []
            seen_contents = set()
            
            logger.info(f"""
            Starting result merger:
            Hybrid results: {len(hybrid_results)}
            Tool results: {len(tool_results)}
            """)
            
            def process_result(result: Dict, source: str):
                content = result.get("content", "")
                # Create a hash of the content to check for duplicates
                if isinstance(content, dict):
                    content_hash = json.dumps(content, sort_keys=True)
                else:
                    content_hash = str(content)
                    
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    # Add source information to track where the result came from
                    result["source_method"] = source
                    merged.append(result)
                    logger.debug(f"Added unique result from {source} (content length: {len(str(content))} chars)")
            
            # Process hybrid results
            for result in hybrid_results:
                process_result(result, "hybrid")
            
            # Process tool results
            for result in tool_results:
                process_result(result, "tool")
            
            logger.info(f"""
            Merge results:
            Total unique documents: {len(merged)}
            Duplicates filtered: {len(hybrid_results) + len(tool_results) - len(merged)}
            """)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            # Return empty list in case of error
            return []