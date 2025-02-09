from typing import List, Dict, Any
import fitz
import re
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

@dataclass
class TextBlock:
    id: int
    text: str
    position: Dict[str, float]
    font: str
    size: float
    block_type: str
    metadata: Dict[str, Any]

class TextProcessor:
    def __init__(self):
        self.logger = logger  # Initialize logger at class level
        self.min_block_length = 10
        try:
            # Try to load the model, download if not present
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.nlp = None

    async def process(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Process text blocks on a page."""
        self.logger.info(f"Processing page {page.number} for text blocks")
        
        # Extract text blocks
        blocks = page.get_text("blocks")
        self.logger.info(f"Found {len(blocks)} raw text blocks on page {page.number}")
        
        processed_blocks = []
        for idx, block in enumerate(blocks):
            try:
                processed_block = self._process_block(block)
                if processed_block:
                    self.logger.info(
                        f"Processed text block {idx + 1}/{len(blocks)} on page {page.number}:\n"
                        f"Position: ({block[0]:.1f}, {block[1]:.1f}, {block[2]:.1f}, {block[3]:.1f})\n"
                        f"Word count: {len(processed_block['text'].split())}\n"
                        f"Content preview: {processed_block['text'][:100]}..."
                    )
                    processed_blocks.append(processed_block)
                
            except Exception as e:
                self.logger.error(
                    f"Error processing text block {idx} on page {page.number}: {str(e)}"
                )
        
        self.logger.info(
            f"Successfully processed {len(processed_blocks)}/{len(blocks)} text blocks on page {page.number}"
        )
        return processed_blocks

    def _extract_raw_blocks(self, page: fitz.Page) -> List[Dict]:
        """Extract raw text blocks with formatting information."""
        blocks = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") == 0:  # Text block
                blocks.append({
                    "text": block.get("text", "").strip(),
                    "bbox": block.get("bbox", [0, 0, 0, 0]),
                    "lines": block.get("lines", []),
                    "font": self._get_block_font(block),
                    "size": self._get_block_size(block)
                })
        return blocks

    async def _process_block(self, block: Dict, idx: int) -> Dict[str, Any]:
        """Process a single text block with NLP and embedding generation."""
        try:
            text = block["text"]
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(text)
            
            # Perform NLP analysis
            nlp_analysis = self._analyze_text(text)
            
            # Determine block type
            block_type = self._determine_block_type(block, nlp_analysis)
            
            return {
                "id": idx,
                "text": text,
                "position": {
                    "x0": block["bbox"][0],
                    "y0": block["bbox"][1],
                    "x1": block["bbox"][2],
                    "y1": block["bbox"][3]
                },
                "font": block["font"],
                "size": block["size"],
                "block_type": block_type,
                "embeddings": embeddings,
                "metadata": {
                    "entities": nlp_analysis["entities"],
                    "keywords": nlp_analysis["keywords"],
                    "sentiment": nlp_analysis["sentiment"],
                    "language": nlp_analysis["language"]
                }
            }
        
        except Exception as e:
            logger.error(f"Error processing block {idx}: {e}")
            return None

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform NLP analysis on text."""
        if not self.nlp:
            return {
                "entities": [],
                "keywords": [],
                "sentiment": 0.0,
                "language": "unknown"
            }

        try:
            doc = self.nlp(text)
            
            # Extract named entities
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
            
            # Extract keywords (noun phrases and important entities)
            keywords = list(set([
                chunk.text for chunk in doc.noun_chunks
            ] + [
                token.text for token in doc
                if token.pos_ in ["NOUN", "PROPN"] and token.is_alpha
            ]))
            
            # Basic sentiment analysis
            sentiment = self._calculate_sentiment(doc)
            
            # Language detection
            language = doc.lang_
            
            return {
                "entities": entities,
                "keywords": keywords[:10],  # Top 10 keywords
                "sentiment": sentiment,
                "language": language
            }
        
        except Exception as e:
            logger.error(f"Error in NLP analysis: {e}")
            return {
                "entities": [],
                "keywords": [],
                "sentiment": 0.0,
                "language": "unknown"
            }

    async def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using BERT."""
        if not self.tokenizer or not self.model:
            return []

        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings[0].cpu().numpy().tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def _determine_block_type(self, block: Dict, nlp_analysis: Dict) -> str:
        """Determine the type of text block based on formatting and content."""
        text = block["text"]
        font_size = block["size"]
        
        # Check for headers
        if font_size > 14:
            return "header"
        
        # Check for lists
        if text.strip().startswith(('•', '-', '*', '1.', '1)', 'a)', 'A.')):
            return "list_item"
        
        # Check for quotes
        if text.strip().startswith('"') and text.strip().endswith('"'):
            return "quote"
        
        # Check for code blocks
        if self._is_code_block(text):
            return "code"
        
        # Default to paragraph
        return "paragraph"

    def _analyze_document_structure(self, blocks: List[Dict]) -> List[Dict]:
        """Analyze the document structure and relationships between blocks."""
        structured_blocks = []
        current_section = None
        
        for block in blocks:
            # Update section tracking
            if block["block_type"] == "header":
                current_section = block["text"]
            
            # Add section information
            block["metadata"]["section"] = current_section
            
            # Add hierarchical information
            if len(structured_blocks) > 0:
                block["metadata"]["previous_block_type"] = structured_blocks[-1]["block_type"]
                block["metadata"]["previous_block_id"] = structured_blocks[-1]["id"]
            
            structured_blocks.append(block)
        
        return structured_blocks

    @staticmethod
    def _is_valid_block(block: Dict) -> bool:
        """Check if a text block is valid for processing."""
        text = block.get("text", "").strip()
        return len(text) > 0 and not text.isspace()

    @staticmethod
    def _get_block_font(block: Dict) -> str:
        """Extract font information from a block."""
        try:
            return block["lines"][0]["spans"][0]["font"]
        except (IndexError, KeyError):
            return "unknown"

    @staticmethod
    def _get_block_size(block: Dict) -> float:
        """Extract font size from a block."""
        try:
            return block["lines"][0]["spans"][0]["size"]
        except (IndexError, KeyError):
            return 0.0

    @staticmethod
    def _calculate_sentiment(doc) -> float:
        """Calculate basic sentiment score."""
        positive_words = set(['good', 'great', 'excellent', 'positive', 'wonderful', 'best', 'amazing'])
        negative_words = set(['bad', 'poor', 'negative', 'terrible', 'worst', 'awful'])
        
        words = [token.text.lower() for token in doc]
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total

    @staticmethod
    def _is_code_block(text: str) -> bool:
        """Detect if text block is likely code."""
        code_indicators = [
            lambda x: bool(re.search(r'(def|class|import|from|return|if|for|while)\s', x)),
            lambda x: bool(re.search(r'[{}\[\]()<>]', x)),
            lambda x: bool(re.search(r'(var|let|const|function)\s', x)),
            lambda x: bool(re.search(r'^\s*(#|//|/\*|\*)', x, re.MULTILINE))
        ]
        
        return any(indicator(text) for indicator in code_indicators)
