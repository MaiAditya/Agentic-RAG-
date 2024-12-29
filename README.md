# Multimodal PDF Pipeline

A robust PDF processing and question-answering system that leverages RAG (Retrieval-Augmented Generation) to provide accurate answers from PDF documents. The system processes text, tables, and images from PDFs to provide comprehensive multimodal analysis.

## Overview

This system allows users to:
1. Upload and process PDF documents with multimodal content
2. Extract and analyze text, tables, and images
3. Store processed content in a vector database (ChromaDB)
4. Query the documents using natural language
5. Receive AI-generated responses based on the document content

## Architecture

### Components

1. **PDF Agent**
   - Handles PDF processing and query answering
   - Uses OpenAI's GPT-3.5-turbo model
   - Implements RAG for accurate responses
   - Manages tool selection and execution for queries

2. **PDF Processor**
   - Extracts content from PDF documents
   - Contains specialized processors:
     - Text Processor: Handles text extraction and analysis
     - Image Processor: Processes and analyzes images
     - Table Processor: Identifies and extracts tabular data

3. **Vector Database (ChromaDB)**
   - Maintains separate collections for text, images, and tables
   - Stores document embeddings
   - Enables semantic search capabilities
   - Retrieves relevant context for queries

### Tools
1. **Document Search Tool**
   - Searches text content in the document
   - Returns top 3 most relevant results

2. **Table Search Tool**
   - Specifically searches tabular data
   - Returns top 2 most relevant table matches

## Implementation Details

### PDF Processing Flow

1. **Document Upload**
   - PDF files are uploaded and processed asynchronously
   - Content is extracted and structured
   - Images are processed with quality checks and filtering
   - Tables are identified and extracted
   - Reference: `PDFAgent.process_pdf()` method

2. **Query Processing**
   - Users submit natural language queries
   - System retrieves relevant context from the vector database
   - AI generates comprehensive answers using the retrieved context
   - Reference: `PDFAgent.answer_query()` method

### Key Features

1. **Intelligent Query Processing**
   - Uses RAG to combine retrieved context with AI generation
   - Limited to 3 most relevant results for focused answers
   - Fallback handling for cases with no relevant information
   - ReAct prompt framework for systematic reasoning

2. **Image Processing**
   - Quality threshold filtering
   - Minimum size requirements
   - Parallel processing with thread pooling
   - Image captioning capabilities
   - Error handling for corrupt or invalid images

3. **Text Analysis**
   - Document structure analysis
   - Font and formatting detection
   - Code block identification
   - Basic sentiment analysis
   - Section and hierarchy tracking

4. **Error Handling**
   - Robust error management throughout the pipeline
   - Detailed logging for debugging and monitoring
   - Graceful fallbacks for processing failures

## Setup and Usage

1. **Environment Setup**
   ```bash
   # Install dependencies using Poetry
   poetry install
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your configurations
   ```

2. **Required Environment Variables**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PORT`: Server port (default: 8000)
   - `HOST`: Host address (default: 0.0.0.0)
   - `CHROMA_DB_PATH`: ChromaDB storage path

## Dependencies

- Python ^3.9
- FastAPI ^0.104.0
- ChromaDB ^0.4.0
- LangChain 0.1.0
- PyMuPDF ^1.23.0
- OpenAI ^1.0.0
- Torch 2.2.2
- Transformers ^4.35.0
- NLTK ^3.9.1
- [See pyproject.toml for complete list]

## API Reference

### PDF Processing
```python
POST /process-pdf
```
- Accepts multipart/form-data with PDF file
- Returns processing status and details

### Query Processing
```python
POST /query
```
- Accepts JSON with query string
- Returns AI-generated response based on document content