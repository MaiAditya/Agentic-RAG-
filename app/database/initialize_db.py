import chromadb
from chromadb.config import Settings
import spacy

def initialize_chroma():
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(
        path="/app/chroma_db",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Create collections if they don't exist
    collections = {}
    collection_names = ["text", "images", "tables"]
    
    for name in collection_names:
        collections[name] = client.get_or_create_collection(name=name)
    
    return client, collections 