import os
from openai import OpenAI
from typing import List


def get_embedding_client():
    """Initialize embedding client using OpenAI-compatible API."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def emb_text(client: OpenAI, text: str, model: str = None) -> List[float]:
    """
    Generate text embeddings using OpenAI-compatible embedding model.
    
    Args:
        client: OpenAI-compatible client
        text: Input text to embed
        model: Embedding model name (default from environment)
    
    Returns:
        List of float values representing the text embedding
    """
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    try:
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}...")
        print(f"Error details: {e}")
        raise


def emb_texts_batch(client: OpenAI, texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batch.
    
    Args:
        client: OpenAI-compatible client
        texts: List of input texts to embed
        model: Embedding model name (default from environment)
    
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    embeddings = []
    
    for text in texts:
        try:
            response = client.embeddings.create(input=text, model=model)
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for text: {text[:50]}...")
            print(f"Error details: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 1536)  # Assuming 1536 dimensions
    
    return embeddings
