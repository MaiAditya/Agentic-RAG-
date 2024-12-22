# Multimodal PDF Pipeline

A robust PDF processing and question-answering system that leverages RAG (Retrieval-Augmented Generation) to provide accurate answers from PDF documents.

## Overview

This system allows users to:
1. Upload and process PDF documents
2. Store extracted content in a vector database
3. Query the documents using natural language
4. Receive AI-generated responses based on the document content

## Architecture

### Components

1. **PDF Agent**
   - Handles PDF processing and query answering
   - Uses OpenAI's GPT-3.5-turbo model
   - Implements RAG for accurate responses

2. **PDF Processor**
   - Extracts content from PDF documents
   - Processes and structures the extracted information

3. **Vector Database (ChromaDB)**
   - Stores document embeddings
   - Enables semantic search capabilities
   - Retrieves relevant context for queries

## Implementation Details

### PDF Processing Flow

1. **Document Upload**
   - PDF files are uploaded and processed asynchronously
   - Content is extracted and structured
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

2. **Error Handling**
   - Robust error management throughout the pipeline
   - Detailed logging for debugging and monitoring

## Setup and Usage

[Add setup instructions here]

## Dependencies

- OpenAI GPT-3.5-turbo
- ChromaDB
- [Add other dependencies]

## Configuration

[Add configuration details here]

## API Reference

### PDF Processing
```python
POST /process-pdf
```
