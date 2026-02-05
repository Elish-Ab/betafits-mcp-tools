"""
SUPABASE RAG MODULE
- This Module exports all the necessary tools and components required for:
- Retrieval in Workflow.
- Creating Embeddings.
- Augmenting Data in the Supabase vector store.
"""

from .embedding import model
from .supabase_client import SUPABASE_CLIENT_
from .supabase_vectorstore import SupabaseRAG


__all__ = ["model", "SUPABASE_CLIENT_", "SupabaseRAG"]
