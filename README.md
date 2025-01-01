# Multimodal PDF Pipeline

A comprehensive PDF processing and question-answering system leveraging RAG (Retrieval-Augmented Generation) for intelligent document analysis and querying. This enterprise-grade solution processes text, tables, and images from PDFs to provide accurate, context-aware responses.

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features

- **Multimodal Processing**
  - Text extraction with formatting preservation
  - Table detection and structured data extraction
  - Image processing with quality analysis
  - Code block identification
  - Document structure analysis

- **Advanced Querying**
  - Natural language question answering
  - Context-aware responses using RAG
  - Multi-document search capabilities
  - Semantic similarity matching

- **Enterprise Ready**
  - Scalable architecture
  - Robust error handling
  - Comprehensive logging
  - API-first design
  - Docker support

## System Requirements

- **Hardware**
  - CPU: 4+ cores recommended
  - RAM: 8GB minimum, 16GB recommended
  - Storage: 10GB+ free space

- **Software**
  - Python 3.9+
  - Docker (optional)
  - CUDA-compatible GPU (optional, for improved performance)

## Installation

### Using Poetry (Recommended)
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yourusername/multimodal-pdf-pipeline
cd multimodal-pdf-pipeline

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Set up environment variables
cp .env.example .env
```

### Using Docker Compose (Production Ready)

1. Create a `docker-compose.yml` file:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_DB_PATH=/app/data/chromadb
      - LOG_LEVEL=INFO
      - MAX_TOKENS=2000
      - TEMPERATURE=0.7
    depends_on:
      - chromadb

  chromadb:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/data

volumes:
  chroma-data:
```

2. Start the services:
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Architecture

### Component Diagram

```plaintext
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│  PDF Agent   │────▶│PDF Processor │
│   Server     │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                           │                      │
                           ▼                      ▼
                    ┌──────────────┐     ┌──────────────┐
                    │   ChromaDB   │     │  OpenAI API  │
                    │              │     │              │
                    └──────────────┘     └──────────────┘
```

### Core Components

1. **PDF Agent**

```python
class PDFAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model = OpenAI(model_name=model_name)
        self.processor = PDFProcessor()
        self.db = ChromaDB()
    
    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process PDF and store in vector database
        """
        try:
            # Extract content
            content = await self.processor.extract_content(file_path)
            
            # Store in vector database
            doc_id = await self.db.store(content)
            
            return {"status": "success", "doc_id": doc_id}
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise PDFProcessingError(str(e))
```

2. **PDF Processor**

```python
class PDFProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.table_processor = TableProcessor()
    
    async def extract_content(self, file_path: str) -> Dict[str, Any]:
        """
        Extract and process PDF content
        """
        doc = fitz.open(file_path)
        content = {
            "text": await self.text_processor.process(doc),
            "images": await self.image_processor.process(doc),
            "tables": await self.table_processor.process(doc)
        }
        return content
```

## Usage Guide
### Starting the Server

```bash
# Development
uvicorn app.main:app --reload

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Processing PDFs

```python
import requests

# Upload PDF
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/process-pdf', files=files)
doc_id = response.json()['doc_id']

# Query the document
query = {
    "text": "What are the main findings in the document?",
    "doc_id": doc_id
}
response = requests.post('http://localhost:8000/query', json=query)
print(response.json()['answer'])
```

## API Reference

### Core Endpoints

1. **Process PDF Document**
```http
POST /process-pdf
Content-Type: multipart/form-data

Parameters:
- file: PDF file (required)

Response:
{
    "status": "success",
    "message": "PDF processed successfully",
    "file_size": integer,
    "details": {
        // Processing details
    }
}
```

2. **Query Documents**
```http
POST /query
Content-Type: application/json

{
    "query": "string"
}

Response:
{
    "status": "success",
    "response": {
        // Query response
    }
}
```

3. **Health Check**
```http
GET /health

Response:
{
    "status": "healthy",
    "chroma_db": {
        // Database statistics
    }
}
```

4. **Collection Statistics**
```http
GET /stats

Response:
{
    "status": "success",
    "stats": {
        // Collection statistics
    }
}
```

5. **Debug Document Processing**
```http
POST /debug-document
Content-Type: multipart/form-data

Parameters:
- file: PDF file (required)

Response:
{
    "status": "success",
    "document_structure": {
        "num_pages": integer,
        "sample_page": object,
        "metadata": object,
        "content_types": {
            "has_text": boolean,
            "has_images": boolean,
            "has_tables": boolean
        }
    }
}
```

### Development Endpoints

6. **Test Document Addition**
```http
POST /test-add

Response:
{
    "status": "success",
    "message": "Test documents added",
    "stats": {
        // Collection statistics
    }
}
```

7. **Collection Information**
```http
GET /collection-info

Response:
{
    "status": "success",
    "collections": {
        // Collection details
    }
}
```

8. **Test Processing Pipeline**
```http
POST /test-processing
Content-Type: multipart/form-data

Parameters:
- file: PDF file (required)

Response:
{
    "status": "success",
    "extraction": {
        "page_count": integer,
        "document_count": integer,
        "documents": array
    },
    "collection_stats": object
}
```

## Configuration

### Environment Variables
```bash
# .env
OPENAI_API_KEY=your_api_key
CHROMA_DB_PATH=./data/chromadb
LOG_LEVEL=INFO
MAX_TOKENS=2000
TEMPERATURE=0.7
HOST=0.0.0.0
PORT=8000
```

### Poetry Configuration
```toml
# pyproject.toml
[tool.poetry]
name = "multimodal-pdf-pipeline"
version = "1.0.0"
description = "A comprehensive PDF processing and question-answering system"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "^0.68.0"
uvicorn = "^0.15.0"
python-multipart = "^0.0.5"
nltk = "^3.6.3"
chromadb = "^0.3.0"
loguru = "^0.5.3"
pydantic = "^1.8.2"
python-dotenv = "^0.19.0"

[tool.poetry.dev-dependencies]
pytest = "^6.2.5"
black = "^21.7b0"
isort = "^5.9.3"
flake8 = "^3.9.2"
```

## Development

### Project Structure

```plaintext
pdf-pipeline/
├── app/
│   ├── main.py
│   ├── agents/
│   ├── processors/
│   └── utils/
├── tests/
├── docs/
├── docker/
├── requirements.txt
└── README.md
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## Troubleshooting

### Common Issues

1. **PDF Processing Fails**
   - Check PDF file permissions
   - Verify file isn't corrupted
   - Ensure sufficient memory

2. **Slow Query Response**
   - Check vector database index
   - Verify chunk size configuration
   - Monitor API rate limits

### Logging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.