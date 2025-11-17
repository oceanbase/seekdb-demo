import pyseekdb
from pyseekdb import HNSWConfiguration
from typing import List, Dict, Any, Tuple

# Simple cache
_client_cache = {}


def get_seekdb_client(db_dir: str = "./seekdb_rag", db_name: str = "test"):
    """Initialize SeekDB client (embedded mode)."""
    cache_key = (db_dir, db_name)
    if cache_key not in _client_cache:
        print(f"Connecting to SeekDB: path={db_dir}, database={db_name}")
        _client_cache[cache_key] = pyseekdb.Client(path=db_dir, database=db_name)
        print("SeekDB client connected successfully")
    return _client_cache[cache_key]


def get_collection(client, collection_name: str = "embeddings", 
                  embedding_dim: int = 1536, drop_if_exists: bool = True):
    """Get or create a collection."""
    if drop_if_exists and client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists, deleting old data...")
        client.delete_collection(collection_name)
    
    config = HNSWConfiguration(dimension=embedding_dim, distance='l2')
    collection = client.get_or_create_collection(
        name=collection_name,
        configuration=config,
        embedding_function=None
    )
    
    print(f"Collection '{collection_name}' ready (dimension: {embedding_dim})")
    return collection


def insert_embeddings(collection, data: List[Dict[str, Any]]):
    """Insert embeddings data into collection."""
    try:
        collection.add(
            ids=[f"{item['source_file']}_{item.get('chunk_index', 0)}" for item in data],
            documents=[item['text'] for item in data],
            embeddings=[item['embedding'] for item in data],
            metadatas=[{'source_file': item['source_file'], 
                       'chunk_index': item.get('chunk_index', 0)} for item in data]
        )
        print(f"Inserted {len(data)} embeddings successfully")
    except Exception as e:
        print(f"Error inserting embeddings: {e}")
        raise


def search_similar_embeddings(
    collection,
    query_embedding: List[float],
    limit: int = 3
) -> List[Tuple[str, float, str, float]]:
    """Search for similar embeddings."""
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        
        return [
            (
                results["documents"][0][i],
                1.0 / (1.0 + results["distances"][0][i]),
                results["metadatas"][0][i].get('source_file', '') if results["metadatas"][0][i] else '',
                results["distances"][0][i]
            )
            for i in range(len(results["ids"][0]))
        ]
    except Exception as e:
        print(f"Error searching embeddings: {e}")
        raise


def get_database_stats(collection) -> Dict[str, Any]:
    """Get statistics about the collection."""
    try:
        results = collection.get(limit=10000, include=["metadatas"])
        ids = results.get('ids', []) if isinstance(results, dict) else []
        metadatas = results.get('metadatas', []) if isinstance(results, dict) else []
        
        unique_files = {m.get('source_file') for m in metadatas if m and m.get('source_file')}
        
        return {
            "total_embeddings": len(ids),
            "unique_source_files": len(unique_files)
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return {"total_embeddings": 0, "unique_source_files": 0}

