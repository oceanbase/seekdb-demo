import sys
import os
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Iterator

from encoder import emb_text, get_embedding_client
from seekdb_utils import get_seekdb_client, get_collection, insert_embeddings
from dotenv import load_dotenv

load_dotenv()

MAX_TEXT_LENGTH = 8000
BATCH_SIZE = 10


def safe_emb_text(client, text: str):
    """Generate embedding, return None if failed."""
    try:
        text = text.strip()
        if not text:
            return None
        
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        
        return emb_text(client, text)
    except Exception:
        return None


def load_markdown_files(data_dir: str) -> Dict[str, List[str]]:
    """Load markdown files and split into chunks by headers."""
    text_dict = {}
    files = list(glob(os.path.join(data_dir, "**/*.md"), recursive=True))
    
    print(f"Loading {len(files)} markdown files...")
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = [c.strip() for c in f.read().split("# ") if c.strip()]
                if chunks:
                    text_dict[file_path] = chunks
        except Exception:
            continue
    
    return text_dict


def generate_embeddings(text_dict: Dict[str, List[str]], embedding_client) -> Iterator[dict]:
    """Generate embeddings for all text chunks."""
    total_chunks = sum(len(chunks) for chunks in text_dict.values())
    
    with tqdm(total=total_chunks, desc="Generating embeddings") as pbar:
        for filepath, chunks in text_dict.items():
            for idx, chunk_text in enumerate(chunks):
                pbar.update(1)
                
                if not chunk_text.strip():
                    continue
                
                embedding = safe_emb_text(embedding_client, chunk_text)
                if embedding:
                    yield {
                        "text": chunk_text,
                        "embedding": embedding,
                        "source_file": filepath,
                        "chunk_index": idx
                    }


def process_and_insert_data(
    data_dir: str, 
    db_dir: str = "./seekdb_rag", 
    db_name: str = "test", 
    collection_name: str = "embeddings"
):
    """Process text data and insert embeddings into SeekDB."""
    
    # Initialize clients
    print("Initializing clients...")
    embedding_client = get_embedding_client()
    seekdb_client = get_seekdb_client(db_dir=db_dir, db_name=db_name)
    
    # Load documents
    text_dict = load_markdown_files(data_dir)
    if not text_dict:
        print("❌ No documents found.")
        return
    
    total_chunks = sum(len(c) for c in text_dict.values())
    print(f"Loaded {len(text_dict)} files with {total_chunks} chunks")
    
    # Process embeddings and detect dimension from first one
    print("Processing embeddings...")
    batch = []
    processed = 0
    embedding_dim = None
    collection = None
    
    for data in generate_embeddings(text_dict, embedding_client):
        # Detect dimension from first embedding
        if embedding_dim is None:
            embedding_dim = len(data["embedding"])
            print(f"Embedding dimension: {embedding_dim}")
            
            # Create collection after we know the dimension
            collection = get_collection(
                seekdb_client, 
                collection_name=collection_name, 
                embedding_dim=embedding_dim,
                drop_if_exists=True
            )
        
        batch.append(data)
        
        # Insert in batches
        if len(batch) >= BATCH_SIZE:
            try:
                insert_embeddings(collection, batch)
                processed += len(batch)
            except Exception as e:
                print(f"\n⚠️  Error inserting batch: {e}")
            finally:
                batch = []
    
    # Insert remaining data
    if batch:
        try:
            insert_embeddings(collection, batch)
            processed += len(batch)
        except Exception as e:
            print(f"\n⚠️  Error inserting final batch: {e}")
    
    if embedding_dim is None:
        print("❌ Failed to generate any embeddings.")
        return
    
    print(f"\n✅ Successfully processed {processed} chunks")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python seekdb_insert.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    # Validate inputs
    if not os.path.exists(data_dir):
        print(f"❌ Directory '{data_dir}' not found.")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment.")
        sys.exit(1)
    
    process_and_insert_data(data_dir)


if __name__ == "__main__":
    main()
