from typing import Optional, List, Dict
from supabase_rag import SupabaseRAG
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from supabase_rag.supabase_client import SUPABASE_CLIENT_
from supabase_rag.embedding import model as embedding_model

from uuid import uuid4



class StoreToRAGInput(BaseModel):
    texts: List[str] = Field(description="List of text documents to store.")
    metadatas: Optional[List[Dict]] = Field(
        default=None,
        description="Optional list of metadata dictionaries corresponding to each text.",
    )
    workflow_run_id: Optional[str] = Field(
        default=None,
        description="Unique run identifier added to metadata for retrieving inserted Supabase IDs.",
    )


class QueryRAGInput(BaseModel):
    query_text: str = Field(description="The query string.")
    k: int = Field(default=5, description="Number of results to return.")
    filter: Optional[Dict] = Field(
        default=None,
        description='Optional metadata filter (e.g., {"source": "docs"}).',
    )


def store_to_rag_(
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    workflow_run_id: Optional[str] = None,
) -> Dict:
    """Store texts into the Supabase RAG system and return inserted row IDs.

    The function augments provided metadatas with a stable `seq_idx` if not present.
    A `workflow_run_id` is required to safely query back the inserted rows in order.

    Returns:
        dict: {"count": int, "ids": [str], "workflow_run_id": str}
    """
    rag = SupabaseRAG()

    # Ensure workflow_run_id
    if not workflow_run_id:
        workflow_run_id = str(uuid4())

    # Normalize metadatas list length
    metadatas = metadatas or [{} for _ in texts]
    if len(metadatas) != len(texts):
        raise ValueError("Length of metadatas must match length of texts")

    # Inject seq_idx + workflow_run_id into each metadata dict (non-destructive)
    for idx, md in enumerate(metadatas):
        if md is None:
            md = {}
        md.setdefault("seq_idx", idx)
        md.setdefault("workflow_run_id", workflow_run_id)
        metadatas[idx] = md

    # Store via LangChain helper
    rag.from_texts(texts=texts, metadatas=metadatas)

    # Query back inserted rows ordered by seq_idx
    # Supabase filter on metadata->>workflow_run_id; order metadata->>seq_idx ASC
    try:
        response = (
            SUPABASE_CLIENT_.table(rag.table_name)
            .select("id, metadata")
            .eq("metadata->>workflow_run_id", workflow_run_id)
            .order("metadata->>seq_idx", desc=False)
            .execute()
        )
        rows = response.data or []
        # Validate ordering
        ids: List[str] = []
        for row in rows:
            ids.append(row.get("id"))
        result = {
            "count": len(texts),
            "ids": ids,
            "workflow_run_id": workflow_run_id,
        }
        # Basic integrity check
        if len(ids) != len(texts):
            result["warning"] = (
                f"Expected {len(texts)} IDs, retrieved {len(ids)}; partial mapping only."
            )
        return result
    except Exception as e:
        return {
            "count": len(texts),
            "ids": [],
            "workflow_run_id": workflow_run_id,
            "error": f"Failed retrieving IDs: {e}",
        }


def query_rag_(
    query_text: str, k: int = 5, filter: Optional[Dict] = None
) -> List[Dict]:
    """
    Query the Supabase RAG system.

    Args:
        query_text: The query string
        k: Number of results to return
        filter: Optional metadata filter (e.g., {"source": "docs"})

    Returns:
        List of retrieved documents with scores
    """
    rag = SupabaseRAG()
    results = rag.query_rag(query_text=query_text, k=k, filter=filter)
    return results


store_to_rag = StructuredTool.from_function(
    func=store_to_rag_,
    name="store_to_rag",
    description="Stores text documents into Supabase RAG and returns inserted row IDs.",
    return_direct=True,
    args_schema=StoreToRAGInput,
)
query_rag = StructuredTool.from_function(
    func=query_rag_,
    name="query_rag",
    description="Queries the Supabase RAG system and retrieves relevant documents.",
    return_direct=True,
    args_schema=QueryRAGInput,
)

if __name__ == "__main__":
    print(store_to_rag.args_schema.model_json_schema())
    print(query_rag.args_schema.model_json_schema())

    print("Dummy query to RAG")
    results = query_rag.invoke({"query_text": "What is PCF?", "k": 3})
    print(f"Query Results: {results}")
