from langchain_core.vectorstores import InMemoryVectorStore
from supabase_rag import model
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from .fetch_pcf_table import fetch_pcf_table_tool
from datetime import datetime
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from models.gemini_client import gemini
#from models.open_router_client import open_router
import re
import math
import json
import os
from collections import Counter

# Load prompts from config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    PROMPTS = CONFIG.get("prompts", {})

# Initialize vector store
vector_store = InMemoryVectorStore(embedding=model)

# BM25 parameters (standard values)
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# BM25 index (built once during embedding creation)
bm25_state = {
    "idf": {},  # term -> inverse document frequency
    "doc_lengths": {},  # doc_id -> length
    "avgdl": 0,  # average document length
    "doc_vectors": {},  # doc_id -> term frequencies
}


def extract_keywords(text: str) -> List[str]:
    """
    Extract important keywords from text for keyword matching boost.

    Args:
        text: Input text

    Returns:
        List of important keywords
    """
    # Remove common words
    stopwords = {
        "the",
        "is",
        "at",
        "which",
        "on",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "with",
        "to",
        "for",
        "of",
        "as",
        "by",
        "this",
        "that",
        "from",
        "are",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "it",
        "its",
        "we",
        "they",
        "you",
        "he",
        "she",
    }

    # Extract words (alphanumeric + hyphens)
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*\b", text.lower())

    # Filter out stopwords and short words
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    # Get unique keywords while preserving some frequency info
    keyword_counts = {}
    for kw in keywords:
        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    # Return keywords sorted by frequency
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:20]]  # Top 20 keywords


def tokenize(text: str) -> List[str]:
    """Fast tokenization for BM25."""
    return re.findall(r"\b[a-z][a-z0-9-]*\b", text.lower())


def build_bm25_index(documents: List[Document]) -> None:
    """
    Build BM25 index once during embedding creation.
    This is fast - only runs once, not at query time.
    """
    global bm25_state

    doc_count = len(documents)
    term_doc_count = Counter()  # How many docs contain each term
    total_length = 0

    # Pass 1: Count term occurrences and doc lengths
    for doc in documents:
        doc_id = doc.metadata.get("id")
        if not doc_id:
            print(f"⚠️ Warning: Document missing ID in metadata: {doc.metadata}")
            continue

        tokens = tokenize(doc.page_content)
        doc_length = len(tokens)

        bm25_state["doc_lengths"][doc_id] = doc_length
        bm25_state["doc_vectors"][doc_id] = Counter(tokens)
        total_length += doc_length

        # Count unique terms per document
        for term in set(tokens):
            term_doc_count[term] += 1

    # Calculate average document length
    bm25_state["avgdl"] = total_length / doc_count if doc_count > 0 else 0

    # Calculate IDF for each term
    for term, df in term_doc_count.items():
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        bm25_state["idf"][term] = idf

    # Debug: Show sample of indexed IDs
    sample_ids = list(bm25_state["doc_vectors"].keys())[:3]
    print(f"   Sample indexed IDs: {sample_ids}")


def calculate_bm25_score(query_text: str, doc_id: str) -> float:
    """
    Calculate BM25 score for a document.
    Super fast - just math operations on pre-computed index.

    Args:
        query_text: Query text
        doc_id: Document ID

    Returns:
        BM25 score (typically 0-10 range, normalized to 0-0.3 for boosting)
    """
    # Debug: Check if index is populated
    if not bm25_state["doc_vectors"]:
        print(f"⚠️ BM25 index is empty!")
        return 0.0

    if doc_id not in bm25_state["doc_vectors"]:
        # Debug: Show what IDs we're looking for vs what we have
        available_ids = list(bm25_state["doc_vectors"].keys())[:3]
        print(
            f"⚠️ Doc ID '{doc_id}' not in BM25 index. Available samples: {available_ids}"
        )
        return 0.0

    query_terms = tokenize(query_text)
    doc_vector = bm25_state["doc_vectors"][doc_id]
    doc_length = bm25_state["doc_lengths"][doc_id]
    avgdl = bm25_state["avgdl"]

    score = 0.0
    for term in query_terms:
        if term not in doc_vector:
            continue

        tf = doc_vector[term]
        idf = bm25_state["idf"].get(term, 0)

        # BM25 formula
        numerator = tf * (BM25_K1 + 1)
        denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_length / avgdl))
        score += idf * (numerator / denominator)

    # Normalize to 0-0.3 range for boosting (typical BM25 scores are 0-10)
    normalized_score = min(score / 10.0 * 0.3, 0.3)
    return normalized_score


def calculate_keyword_boost(query_keywords: List[str], doc_content: str) -> float:
    """
    Calculate a boost score based on keyword matches.

    Args:
        query_keywords: Keywords from the query
        doc_content: Document content to search

    Returns:
        Boost score (0.0 to 0.3) to add to similarity score
    """
    if not query_keywords:
        return 0.0

    doc_lower = doc_content.lower()
    matches = sum(1 for kw in query_keywords if kw in doc_lower)

    # Convert to boost score (max 0.3 boost)
    match_ratio = matches / len(query_keywords)
    boost = match_ratio * 0.3

    return boost


def extract_key_topics_from_transcript(transcript: str) -> str:
    """
    Extract key topics, project names, and action items from a transcript.
    This helps focus the semantic search on relevant content.

    Args:
        transcript: Raw meeting transcript or text

    Returns:
        Condensed summary with key topics and relevant terms
    """
    # Load prompt template from config
    prompt_config = PROMPTS.get("extract_key_topics", {})
    prompt_template = prompt_config.get("template", "")

    # Format prompt with transcript (truncate to 4000 chars)
    prompt = prompt_template.format(transcript=transcript[:4000])

    try:
        response = gemini.invoke(prompt)
        #response = open_router.invoke(prompt)
        summary = response.content.strip()
        print("\n🎯 Extracted Key Topics from Transcript:")
        print(f"{'=' * 80}")
        print(summary)
        print(f"{'=' * 80}\n")
        return summary
    except Exception as e:
        print(f"⚠️ Error extracting topics: {e}")
        print("📝 Using first 500 chars of transcript as fallback")
        # Fallback: use first part of transcript
        return transcript[:500]


def rerank_with_llm(
    query_summary: str, candidates: List[Tuple[Document, float]], top_k: int = 10
) -> List[Dict]:
    """
    Re-rank search results using LLM to assess true relevance.
    This provides more accurate scoring than pure embedding similarity.

    Args:
        query_summary: The extracted key topics from the query
        candidates: List of (Document, embedding_score) tuples from vector search
        top_k: Number of top results to return after re-ranking

    Returns:
        List of re-ranked PCF dictionaries with LLM-assigned relevance scores (0.0-1.0)
    """
    if not candidates:
        return []

    print(
        f"\n🤖 Re-ranking {len(candidates)} candidates with LLM for better accuracy..."
    )
    print(f"{'=' * 80}")

    # Build candidate list for LLM evaluation
    candidate_texts = []
    for idx, (doc, emb_score) in enumerate(candidates, 1):
        candidate_texts.append(
            f"""
[Candidate {idx}]
ID: {doc.metadata.get("id")}
Name: {doc.metadata.get("name")}
Type: {doc.metadata.get("pcf_type")}
Description: {doc.page_content[:500]}
"""
        )

    # Load prompt template from config
    prompt_config = PROMPTS.get("rerank_with_llm", {})
    prompt_template = prompt_config.get("template", "")

    # Format prompt with variables
    prompt = prompt_template.format(
        query_summary=query_summary, candidate_texts="".join(candidate_texts)
    )

    try:
        response = gemini.invoke(prompt)
        #response = open_router.invoke(prompt)
        import json

        # Extract JSON from response
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        llm_scores = json.loads(content)

        # Create a mapping of id -> llm score
        id_to_llm_score = {item["id"]: item for item in llm_scores}

        # Re-rank candidates based on LLM scores + BM25 boost
        reranked = []
        for doc, emb_score in candidates:
            pcf_id = doc.metadata.get("id")
            llm_data = id_to_llm_score.get(
                pcf_id,
                {
                    "score": emb_score,
                    "reason": "LLM scoring failed, using embedding score",
                },
            )

            # Calculate BM25 boost (fast lookup from pre-computed index)
            bm25_boost = calculate_bm25_score(query_summary, pcf_id)

            # Combine LLM score with BM25 boost (cap at 1.0)
            final_score = min(llm_data["score"] + bm25_boost, 1.0)

            reranked.append(
                {
                    "pcf_id": pcf_id,
                    "name": doc.metadata.get("name", "Unknown"),
                    "pcf_type": doc.metadata.get("pcf_type", "Unknown"),
                    "department": doc.metadata.get("department", ""),
                    "stage": doc.metadata.get("stage", ""),
                    "similarity_score": float(final_score),
                    "llm_score": float(llm_data["score"]),
                    "bm25_boost": float(bm25_boost),
                    "llm_reason": llm_data.get("reason", ""),
                    "embedding_score": float(emb_score),
                }
            )

        # Sort by LLM score (descending)
        reranked.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Limit to top_k
        reranked = reranked[:top_k]

        print(f"✓ Re-ranked {len(reranked)} results with LLM scores")
        print(f"{'=' * 80}\n")

        return reranked

    except Exception as e:
        print(f"⚠️ LLM re-ranking failed: {e}")
        print("📝 Falling back to embedding scores only\n")

        # Fallback: use embedding scores
        fallback = []
        for doc, emb_score in candidates[:top_k]:
            fallback.append(
                {
                    "pcf_id": doc.metadata.get("id"),
                    "name": doc.metadata.get("name", "Unknown"),
                    "pcf_type": doc.metadata.get("pcf_type", "Unknown"),
                    "department": doc.metadata.get("department", ""),
                    "stage": doc.metadata.get("stage", ""),
                    "similarity_score": float(emb_score),
                    "llm_reason": "LLM re-ranking unavailable",
                    "embedding_score": float(emb_score),
                }
            )
        return fallback


def format_pcf_record_text(record: Dict) -> str:
    """
    Format a PCF record into a semantically meaningful text representation.

    Creates a natural, narrative-style description that captures the essence
    of the PCF record for better semantic matching. Prioritizes:
    - Clear identification and categorization
    - Rich contextual descriptions
    - Related entities and relationships
    - Action items and deliverables
    """
    fields = record.get("fields", {})

    # Core identifying information
    name = fields.get("Name", "")
    pcf_type = fields.get("PCF", "")  # Project/Component/Feature
    pcf_formula = fields.get("P/C/F", "")  # Combined identifier

    # Categorization
    department = fields.get("Department", "")
    stage = fields.get("Stage", "")
    status = fields.get("Status", "")
    type_tags = fields.get("Type", [])

    # Rich text content (most important for semantic matching)
    objective = fields.get("Objective", "")
    pcf_context = fields.get("PCF Context", "")
    approach = fields.get("Approach", "")
    notes = fields.get("Notes", "")
    suggested_steps = fields.get("Suggested Steps", "")
    milestones = fields.get("Milestones", "")
    description = fields.get("Description", "")

    # Expanded linked records
    projects = fields.get("Projects", [])
    components = fields.get("Components", [])
    features = fields.get("Features", [])
    team_members = fields.get("Team Members", [])
    tools = fields.get("Tools", [])

    # Build a natural, flowing narrative
    text_parts = []

    # === INTRODUCTION ===
    # Start with a clear statement about what this is
    intro_parts = []
    if name:
        if pcf_type:
            intro_parts.append(f"This is a {pcf_type} called '{name}'")
        else:
            intro_parts.append(f"This is '{name}'")
    elif pcf_formula:
        intro_parts.append(f"This is {pcf_formula}")

    # Add categorization naturally
    if department:
        intro_parts.append(f"in the {department} department")
    if status:
        intro_parts.append(f"with status: {status}")
    if stage:
        intro_parts.append(f"currently at {stage} stage")

    if intro_parts:
        text_parts.append(". ".join(intro_parts) + ".")

    # Add type tags if present
    if type_tags:
        if isinstance(type_tags, list) and type_tags:
            text_parts.append(f"It is categorized as: {', '.join(type_tags)}.")
        elif isinstance(type_tags, str):
            text_parts.append(f"It is categorized as: {type_tags}.")

    # === DESCRIPTION & OBJECTIVE ===
    # These are the most important for semantic matching
    if description:
        text_parts.append(f"\nDescription: {description}")

    if objective:
        text_parts.append(f"\nObjective: {objective}")

    # === CONTEXT ===
    # Provide background information
    if pcf_context:
        text_parts.append(f"\nBackground and Context: {pcf_context}")

    # === APPROACH & IMPLEMENTATION ===
    if approach:
        text_parts.append(f"\nImplementation Approach: {approach}")

    if suggested_steps:
        text_parts.append(f"\nSuggested Steps: {suggested_steps}")

    if milestones:
        text_parts.append(f"\nKey Milestones: {milestones}")

    # === RELATIONSHIPS ===
    # Add information about linked entities
    relationship_parts = []

    if projects:
        project_info = []
        for p in projects:
            p_name = p.get("Name", "")
            p_desc = p.get("Description", "")
            if p_name:
                if p_desc:
                    project_info.append(f"{p_name} ({p_desc})")
                else:
                    project_info.append(p_name)
        if project_info:
            relationship_parts.append(f"Related Projects: {', '.join(project_info)}")

    if components:
        component_info = []
        for c in components:
            c_name = c.get("Name", "")
            c_desc = c.get("Description", "")
            if c_name:
                if c_desc:
                    component_info.append(f"{c_name} ({c_desc})")
                else:
                    component_info.append(c_name)
        if component_info:
            relationship_parts.append(
                f"Related Components: {', '.join(component_info)}"
            )

    if features:
        feature_info = []
        for f in features:
            f_name = f.get("Name", "")
            f_desc = f.get("Description", "")
            if f_name:
                if f_desc:
                    feature_info.append(f"{f_name} ({f_desc})")
                else:
                    feature_info.append(f_name)
        if feature_info:
            relationship_parts.append(f"Related Features: {', '.join(feature_info)}")

    if relationship_parts:
        text_parts.append("\n" + ". ".join(relationship_parts) + ".")

    # === TEAM & TOOLS ===
    team_parts = []

    if team_members:
        member_names = [t.get("Name", "") for t in team_members if t.get("Name")]
        if member_names:
            team_parts.append(f"Team Members: {', '.join(member_names)}")

    if tools:
        tool_info = []
        for t in tools:
            t_name = t.get("Name", "")
            t_desc = t.get("Description", "")
            if t_name:
                if t_desc:
                    tool_info.append(f"{t_name} ({t_desc})")
                else:
                    tool_info.append(t_name)
        if tool_info:
            team_parts.append(f"Tools and Technologies: {', '.join(tool_info)}")

    if team_parts:
        text_parts.append("\n" + ". ".join(team_parts) + ".")

    # === ADDITIONAL NOTES ===
    if notes:
        text_parts.append(f"\nAdditional Notes: {notes}")

    return "\n".join(text_parts)


def create_pcf_table_embeddings(records: List[Dict]) -> None:
    """
    Create embeddings of PCF records - each record as a single document.
    No chunking to preserve semantic coherence of each PCF record.

    Args:
        records: List of records from the PCF Table (Airtable format)
    """
    documents = []
    skipped_count = 0

    for record in records:
        record_id = record.get("id")
        created_at = record.get("createdTime")
        fields = record.get("fields", {})

        # Skip records with no meaningful content
        if not fields:
            skipped_count += 1
            continue

        # Format the record into searchable text
        formatted_text = format_pcf_record_text(record)

        # Skip if text is too short (likely incomplete record)
        if len(formatted_text.strip()) < 20:
            skipped_count += 1
            continue

        # Create comprehensive metadata
        metadata = {
            "id": record_id,
            "createdAt": created_at,
            "record_type": "pcf_record",
            "name": fields.get("Name", "Unknown"),
            "pcf_type": fields.get("PCF", "Unknown"),
            "department": fields.get("Department", ""),
            "stage": fields.get("Stage", ""),
            "status": fields.get("Status", ""),
            "current": fields.get("Current", False),
        }

        # Create a single document for the entire record (NO CHUNKING)
        doc = Document(page_content=formatted_text, metadata=metadata)
        documents.append(doc)

    # Add all documents to vector store at once
    if documents:
        vector_store.add_documents(documents)
        print(
            f"✓ Added {len(documents)} PCF records as complete documents (no chunking)"
        )
        if skipped_count > 0:
            print(f"  (Skipped {skipped_count} records with insufficient content)")
        print(f"✓ Total documents in store: {len(vector_store.store)}")

        # Build BM25 index (fast, one-time operation)
        print("⚡ Building BM25 index for keyword matching...")
        build_bm25_index(documents)
        print(f"✓ BM25 index ready ({len(bm25_state['idf'])} unique terms)")
    else:
        print("⚠ No documents to add to vector store")


def query_pcf_embeddings(
    query_data: str,
    k: int = 7,
    score_threshold: float = 0.0,
    filter_current: bool = False,
) -> List[Tuple[Document, float]]:
    """
    Query the vector store for relevant PCF records.
    Each result is a complete PCF record (no chunking).

    Args:
        query_data: The search query (e.g., transcript, meeting notes)
        k: Number of top results to return
        score_threshold: Minimum similarity score threshold
        filter_current: If True, only return records marked as "Current"

    Returns:
        List of (Document, score) tuples
    """
    # Get results from vector store
    results = vector_store.similarity_search_with_score(query=query_data, k=k * 2)

    # Apply filters
    filtered_results = []
    for doc, score in results:
        # Filter by score threshold
        if score_threshold > 0 and score < score_threshold:
            continue

        # Filter by current flag if requested
        if filter_current and not doc.metadata.get("current", False):
            continue

        filtered_results.append((doc, score))

    # Limit to k results
    filtered_results = filtered_results[:k]

    # Display results
    print(f"\n{'=' * 80}")
    print(f"Query Results (Top {len(filtered_results)} PCF Records):")
    print(f"{'=' * 80}")

    for idx, (doc, score) in enumerate(filtered_results, 1):
        print(f"\n[{idx}] Similarity Score: {score:.4f}")
        print(f"    PCF ID: {doc.metadata.get('id', 'N/A')}")
        print(f"    Name: {doc.metadata.get('name', 'N/A')}")
        print(f"    Type: {doc.metadata.get('pcf_type', 'N/A')}")
        print(f"    Department: {doc.metadata.get('department', 'N/A')}")
        print(f"    Stage: {doc.metadata.get('stage', 'N/A')}")
        print(f"    Status: {doc.metadata.get('status', 'N/A')}")
        print(f"    Content Preview: {doc.page_content[:200].replace(chr(10), ' ')}...")

    print(f"{'=' * 80}\n")

    return filtered_results


class CreateEmbeddingsInput(BaseModel):
    """Input schema for creating PCF table embeddings."""

    records: List[Dict] = Field(
        description="List of records from the PCF Table (Airtable format)"
    )


class MapRelevantPCFsInput(BaseModel):
    """Input schema for mapping relevant PCFs."""

    transcript: str = Field(
        description="The transcript or text to match against PCF records"
    )
    top_k: int = Field(
        default=10, description="Number of top relevant PCFs to return (1-20)"
    )
    score_threshold: float = Field(
        default=0.0, description="Minimum similarity score threshold (0.0-1.0)"
    )
    filter_current: bool = Field(
        default=False, description="Only return PCFs marked as 'Current'"
    )


class RelevantPCF(BaseModel):
    """Schema for a single relevant PCF result."""

    pcf_id: str = Field(description="Airtable record ID")
    name: str = Field(description="PCF name")
    pcf_type: str = Field(description="Type: Project, Component, or Feature")
    department: str = Field(description="Department")
    stage: str = Field(description="Current stage")
    similarity_score: float = Field(description="Similarity score (higher is better)")


class MapRelevantPCFsOutput(BaseModel):
    """Output schema for mapped PCFs."""

    relevant_pcfs: List[RelevantPCF] = Field(
        description="List of relevant PCF records with details"
    )
    total_found: int = Field(description="Total number of relevant PCFs found")


def map_relevant_pcfs(
    transcript: str,
    top_k: int = 10,
    score_threshold: float = 0.0,
    filter_current: bool = False,
) -> Dict:
    """
    Map the most relevant PCF records for a given transcript.

    This function finds the top_k most relevant PCF records based on
    semantic similarity to the provided transcript or text.
    Each result is a complete PCF record (no chunking/deduplication needed).

    Args:
        transcript: The input text to search against
        top_k: Number of top results to return (default: 10)
        score_threshold: Minimum similarity score (default: 0.0)
        filter_current: Only return PCFs marked as "Current" (default: False)

    Returns:
        Dictionary containing:
        - relevant_pcfs: List of PCF objects with id, name, type, etc.
        - total_found: Total number of relevant PCFs found
    """
    print(f"\n{'🔍 MAPPING RELEVANT PCFs':^80}")
    print(f"{'=' * 80}")
    print(f"Original query length: {len(transcript)} characters")
    print(f"Searching for top {top_k} relevant PCFs")
    print(f"Score threshold: {score_threshold}")
    print(f"Filter current only: {filter_current}")
    print(f"{'=' * 80}\n")

    # STEP 1: Extract key topics from long transcripts to improve relevance
    search_query = transcript
    if len(transcript) > 500:
        print("📝 Transcript is long - extracting key topics for better search...")
        search_query = extract_key_topics_from_transcript(transcript)
    else:
        print("📝 Using full transcript for search (short enough)")
        print(f"Query: {transcript[:200]}...\n")

    # STEP 2: Query the vector store with the processed query (get more candidates for re-ranking)
    initial_results = query_pcf_embeddings(
        query_data=search_query,
        k=min(top_k * 3, 30),  # Get 3x candidates for re-ranking (max 30)
        score_threshold=0.0,  # No threshold on initial retrieval
        filter_current=filter_current,
    )

    # STEP 3: Re-rank with LLM for better accuracy
    relevant_pcfs = rerank_with_llm(
        query_summary=search_query, candidates=initial_results, top_k=top_k
    )

    # Apply score threshold after re-ranking
    if score_threshold > 0:
        relevant_pcfs = [
            pcf for pcf in relevant_pcfs if pcf["similarity_score"] >= score_threshold
        ]

    result = {"relevant_pcfs": relevant_pcfs, "total_found": len(relevant_pcfs)}

    print(f"\n{'✓ MAPPING COMPLETE':^80}")
    print(f"{'=' * 80}")
    print(f"Found {result['total_found']} relevant PCF records")
    if relevant_pcfs:
        print("\nTop PCFs (LLM Re-ranked + BM25 Boosted):")
        for i, pcf in enumerate(relevant_pcfs, 1):
            print(f"  {i}. {pcf['name']} ({pcf['pcf_type']})")
            llm_score = pcf.get("llm_score", pcf["similarity_score"])
            bm25_boost = pcf.get("bm25_boost", 0.0)
            print(
                f"     ID: {pcf['pcf_id']} | Final Score: {pcf['similarity_score']:.4f} "
                f"(LLM: {llm_score:.4f} + BM25: {bm25_boost:.4f})"
            )
            print(f"     Department: {pcf['department']} | Stage: {pcf['stage']}")
            if pcf.get("llm_reason"):
                print(f"     Reason: {pcf['llm_reason']}")
    print(f"{'=' * 80}\n")

    return result


# Create the mapping tool for LangChain agents
map_relevant_pcfs_tool = StructuredTool.from_function(
    func=map_relevant_pcfs,
    name="map_relevant_pcfs",
    description=(
        "Map the most relevant PCF (Project/Component/Feature) records for a given transcript "
        "or text. Returns 1-5 (or top_k) relevant PCF IDs with their details and similarity scores. "
        "Use this to find which PCF records are most related to a meeting, conversation, or document."
    ),
    args_schema=MapRelevantPCFsInput,
    return_direct=False,
)


def semantic_matching(
    records: List[Dict],
    query_data: str,
    top_k: int = 10,
    score_threshold: float = 0.0,
    filter_current: bool = False,
) -> Dict:
    """
    Complete semantic matching workflow:
    1. Create embeddings for PCF records
    2. Extract key topics from query (if long)
    3. Query and map relevant PCFs

    Args:
        records: List of PCF records from Airtable
        query_data: Transcript or text to search
        top_k: Number of top results to return (default: 10)
        score_threshold: Minimum similarity score
        filter_current: Only return PCFs marked as "Current"

    Returns:
        Mapping results with relevant PCF details and scores
    """
    print(f"\n{'🚀 STARTING SEMANTIC MATCHING WORKFLOW':^80}")
    print(f"{'=' * 80}")
    print(f"Processing {len(records)} PCF records")
    print(f"{'=' * 80}\n")

    print(f"⏰ Creating Embeddings -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    create_pcf_table_embeddings(records=records)

    print(f"\n⏰ Querying Embeddings -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = map_relevant_pcfs(
        transcript=query_data,
        top_k=top_k,
        score_threshold=score_threshold,
        filter_current=filter_current,
    )

    print(f"\n{'✅ WORKFLOW COMPLETE':^80}")
    print(f"{'=' * 80}\n")

    return results


def clear_vector_store() -> None:
    """Clear all documents from the vector store and BM25 index."""
    global vector_store, bm25_state
    vector_store = InMemoryVectorStore(embedding=model)
    bm25_state = {
        "idf": {},
        "doc_lengths": {},
        "avgdl": 0,
        "doc_vectors": {},
    }
    print("✓ Vector store and BM25 index cleared")


def get_store_stats() -> Dict:
    """Get statistics about the current vector store."""
    total_docs = len(vector_store.store)

    # Count unique PCF IDs (should match total_docs since no chunking)
    unique_pcf_ids = set()
    pcf_type_counts = {}
    department_counts = {}
    stage_counts = {}

    for doc_id, doc_data in vector_store.store.items():
        # Handle both dict and Document objects
        if isinstance(doc_data, dict):
            metadata = doc_data.get("metadata", {})
        else:
            metadata = getattr(doc_data, "metadata", {})

        pcf_id = metadata.get("id")
        if pcf_id:
            unique_pcf_ids.add(pcf_id)

            # Track PCF type distribution
            pcf_type = metadata.get("pcf_type", "Unknown")
            pcf_type_counts[pcf_type] = pcf_type_counts.get(pcf_type, 0) + 1

            # Track department distribution
            department = metadata.get("department", "Unspecified")
            department_counts[department] = department_counts.get(department, 0) + 1

            # Track stage distribution
            stage = metadata.get("stage", "Unspecified")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    return {
        "total_pcf_records": total_docs,
        "unique_pcf_records": len(unique_pcf_ids),
        "note": "No chunking - each document is a complete PCF record",
        "pcf_type_distribution": pcf_type_counts,
        "department_distribution": department_counts,
        "stage_distribution": stage_counts,
    }


if __name__ == "__main__":
    # Example usage - fetch_pcf_table now returns just the records list
    records = fetch_pcf_table_tool.invoke({})

    # Example transcript from a meeting about email automation
    transcript = """
       Megacensus 2.0 Features 
Mon, Mar 24, 2025

6:12 - Unidentified Speaker 
Hey, Matthew. Hey. You in a booth? Yep. I got notifications going off all over the place. Slack's up.

6:53 - Dwayne Gibson 
Yeah, like I said, I set up a MOOC. You can obviously tell a MOOC from my normal workstation and I set it up in my near my PC. I can use both and now I have phone, laptop, and desktop sending me notifications.

7:23 - Ashfaq Ali 
Got it. There's Ali.

7:27 - Matthew Prisco 
How's Ramadan treating you?

7:30 - Ashfaq Ali 
Very good and very busy. Busy? When is the end of Ramadan?

7:36 - Matthew Prisco 
Only one more week.

7:38 - Unidentified Speaker 
One more, okay.

7:40 - Matthew Prisco 
Is it the same thing the whole month or does it, is there extra significance at the beginning, at the end, or just?

7:52 - Ashfaq Ali 
Throughout it's busy, but some days especially, these last 10 days are extra busy. Then there's the Eid, like Christmas for Muslims, the biggest holiday. At the end? At the end. The last day. Then the first of the next month. I see, good.

8:17 - Matthew Prisco 
Okay, well, let's, I guess, set the topic for today. So we want to think about the architecture for mega census version 2.0 So we built right the first the first version Obviously before AI Came to the forefront before we had experience building some of the workflows that now we know how to do So I want to think You know, what parts of this can we change quickly, you know, at this stage, and what might the future of mega census look like, How much of this do we want to run through Google and workbooks, the master sort of idea? Or do we want to have more of that go through Airtable as, you know, newer automations that we're working on? Do we want that to be the panel of how we request a certain format, where it starts, what file gets created? I think we're starting to do more of that through Airtable. And I think specific to the architecture of Megasensus, we sort of have two steps. First step is the input. Conversion and then the output conversions So there may be some things that we should redesign about About both sides of that On the input I think we had designed a few common Formats that we were getting a lot of or that we would expect to still get a lot of and then had a hard code of this goes Yeah, then my understanding once we put it in to this the beta fit standard It's going to create a workbook with every other hard-coded possibility And then Dwayne could pick from that whichever one he wants It's possible we can keep that on If we had 30 different output formats do we really need it to create 30? Possibilities or do we want to just store? In the beta fits format and then have another automation that can give us whichever format we need just as a CSV so that rather than picking It is it's a choice you create one output or you create all of

10:58 - Ashfaq Ali 
them.

10:58 - Matthew Prisco 
I think I don't know exactly I'm not sure I had it in my head or from talking with Dwayne that It just automatically creates every format if we only have five formats.

11:11 - Dwayne Gibson 
That's not a big deal if we had 200 formats now that doesn't doesn't seem necessary so what you're describing is the uh reconvert to destination script And I have not used that much but I think that to the initial step that Matthew's describing does convert to all of the outputs and then I'd have to dig into more why we developed the reconvert to destination. I think maybe we had that if we add something to the census and that step allows us to select the individual output.

11:48 - Ashfaq Ali 
If you change something and then convert it back.

11:52 - Matthew Prisco 
Now, for all of the formats where we have an output format, there's I think two parts of the conversion. One is which fields do we need and the order of those fields. And then any formatting changes. It needs to say employee, or it needs to say EE, or it needs to say self, whatever those are, whatever rules. And picking the columns is easy. Arranging the columns is easy. The question would be, now that we have access to AI, do we, I assume that it was working using Apps Script or Python or whatever it was doing to convert the employee to EE. Do we need to mess with that? And is it a better, a more scalable, reliable solution to use some kind of hard coding so that from the BetaFit standard, we can always get it to reliably convert to whichever formats I think that, you know, my assumption is we don't need AI there. Could AI do it? Sure. But I think if we already have a process to do it on other, with other conversions or transformations, let's stick with that. Now let's talk about the input side. We've got a few input formats where we say if we get it in this one that we get a lot ease, for example. We are going to do a lot of ease so maybe it still makes sense to do a hard conversion from ease to beta fits master And then be able to convert it from there but what I would like to introduce is in addition to the the real standard input files Is a Hatch all where we say we just got this file take the fields that we have Or that are on this file put them into the beta fit standard in the Transformations that are necessary to get it into the beta fit standard So I think that that's obviously a new one that when we would use AI so that if it's not working one of those common inputs that we have a solution that can hopefully get us close. With the extra qualification that we do want to try to avoid putting certain types of fields fully through the LLM or at least to be able to tell our clients we're not putting social security numbers, addresses, even dates of birth. I'm less Protective of dates of birth. So is there a way when we're now designing what I think we can all agree is is the AI version of the input conversion Can we give it just the headers of the file and maybe the first three rows and have the AI? Identify the mapping. So really the prompt that we design is we you know, look at this Here is the the big Fits master mapping all of the possible fields on the beta fits identify what are the fields that we have and which ones map to which and then You know then we can use even Python or or the other automation to do the transformation still but the I think really just to test at first the can we give it just the header or just a few examples and then have it create the mapping. And from the mapping, now we have a way to do the rest of the transformation to the standard without having to put the whole file through.

16:03 - Ashfaq Ali 
It's like we call AI on every file or we call AI when a new format is received.

16:12 - Matthew Prisco 
It would be an option. We're going to start doing a lot of these automated at scale. So if we call a function or an automation that says, go get the latest census from employee navigator for this client, it knows that it's going to need to take that and put it right into mega census and convert it to whatever we want. So the AI wouldn't be in that step if we say, go, grab it, and here's the output that we need. What we would need that is new is the ability to make that request from the control table or the control panel in Airtable. We need to make this table or it'd be like the version that we have currently in Google Sheets, the master will say, census was requested for this client from this platform, it triggered this automation, it brought back this file and blah, blah, whatever that table needs to have. And then it would know whether it needs to incorporate the AI step based on the source of the file.

17:34 - Dwayne Gibson 
Mm-hmm. I just looked up so that we can do that one method But I think there might it might come down or it might be worth investigating which is the easier in terms of workflow So I just searched and apparently there's a library called crypto JS Which would allow us to hard code which columns get encrypted before we send it to the LLM so we could encrypt Social Security date of birth and then process it and then and send it back to Apps Script and it decrypts it. So they never get, it accomplishes the same thing except social security address, whatever we say. I'm just a little bit reluctant on census data.

18:14 - Matthew Prisco 
Well, for two reasons. One is the sensitivity of it. Yes, we could encrypt it, decrypt it. But if we don't need to give it the whole file, I'm trying to think how can we just give it a few rows as a sample and say, you tell us which is which then have a script that can handle any possibility of this mapping just to say now we this is the the date of birth column so make sure that it's in our format with zeros or you know in the right order whatever that needs to look like but we never need to give the LLM the whole file because right now a lot of the groups are 50 people maybe there's 80 people on the census we've got a thousand people on the census the last thing we want to be doing is giving the whole census to the LLM. And even if it makes a mistake one out of a thousand times, if it makes it one out of a hundred times, now we've got a lot of room for error. So I would rather rely on the hard coding once we have the mapping.

19:20 - Ashfaq Ali 
I have two things to add here, two scenarios. One is we call AI for each file and one we call for each format once for each format so let's suppose if we if you're doing it to add more formats source formats so when we have a new format then we give AI that this is the the beta fits version that this is the output desired output what we want and this is the input we know nothing about this input and AI would tell us that this is the code to convert this source into this. So we, the output from the AI would be the code of either Apps Script or Python. And then we can feed it a few rows, like you said, top 10 rows, top five rows. And it can give us the code. And then we use that code from next time onwards.

20:19 - Matthew Prisco 
That's one scenario. I think that in controlled environments, when we're taking it from ease, it from employee navigator, it's our automation, we we can have the repeatable process. What I'm picturing is that in a lot of cases, we're just going to get a file, we don't even necessarily know where it came from Maybe there were other configurations. So it may not even be the standard version, or maybe they added some extra columns, deleted some things, moved it around. So I think we're going to get if the client up loads the census we're not going to have any any way to then you would have to feed the entire file and then you cannot encrypt it because we don't

21:02 - Ashfaq Ali 
know which columns to encrypt encrypt if there are 15 and we know the first two columns are okay with sending the first four the first four rows no but you cannot send the first four rows because it needs to convert the entire thing because now we know nothing. We just know that there's a file and that's it.

21:23 - Matthew Prisco 
No, I think that we could do this manually. Take the LLM out of the equation. We could have somebody look at that file and say we just need to do a mapping. Say column A, B, C, D, E, F, G. We don't know what they are. Maybe the header would give it away. If it says date of birth, now we would know, okay, this is date of birth. Or if we see certain patterns of data, whether we need to look at three rows or 10 rows, a human would be able to create the mapping. I'm saying that we want the process to work where we just give the mapping for one time use, take this file, take column B is first name, C is last name, D is date of birth, G is compensation, enrollment, whatever it is, and create the mapping. What I'm saying is I think we design a process so that AI creates the mapping, which then gets plugged into our hard-coded transformations, and then we can decide, do we need to give it three rows to have a good chance of getting the right mapping? But we could do this even creating the mapping manually, and the rest of the process would still work, but I think we'll have good results with giving a few rows to chat GPT, not worrying about the encoding side of it, encrypting or all of that.

22:51 - Ashfaq Ali 
It's a manual process where we see, okay, we have a new format. We do some manual, random form.

22:58 - Matthew Prisco 
I don't want to think that they're never, I think a lot of them are going to be, we don't know where this came from We are never going to see this again. Is one category versus we are working with this platform and we want to be able to automate it ourself.

23:18 - Unidentified Speaker 
Yeah.

23:18 - Ashfaq Ali 
So someone would open that file, give it to chat GPT and say, give me a transformation of a code that would transform this into beta fits format. Right. And then we have that code and we would run it.

23:32 - Matthew Prisco 
I don't think, I think we need to develop that code ourselves. Can we ask chat GPT, what is the code to do the transformations, yes, but how is that working now? How are we taking, for sure we have the JustWorks input format, how do we transform the JustWorks input format to the BetaFit standard format? That is using Apps Script or Python. It should still use Apps Script or Python. We need to have one code that takes any of these inputs, whether it's one of the ones we know, All that, all it becomes then is the mapping. We don't even need to have hard coding just for just work.

24:16 - Ashfaq Ali 
But that's not, mapping is not the only thing, Matthew. Mapping is just one part of it. And then converting it like, like date of birth formats, like writing the family identifier, right. Using a state and last name and that.

24:33 - Matthew Prisco 
So what I'm saying is, is would we be able to look at the ones that we already have and say, either for the ones we have, we keep it, just use it as is. Or could we even have a common process once we have the mapping? So then the difference would be, we know the mapping from ease to get to the standard. So we would just, we would use that. It could even simplify if this is possible, where we say, here is our transformation step using Python or using whatever it's using. And rather than having all of those transformation steps in a script that's just for just works to beta fits and just for ease to beta fits, if we make it so that that component can take Once we give it the mapping, any input and put it into the beta fit standard, then we can use that same piece of code. So yes, we need to write that code. But then I think the piece that sits in front of it could be, we have the mappings for standard input to beta fits. And now we have this new process where we create the mappings using AI if it's not one of standard inputs?

26:02 - Ashfaq Ali 
Mapping, yes, we can create like a common code, like a common script that we have. It would maybe give the headers to AI and ask him to give me the mappings and then use that return mapping and then do the rest of the thing within our script. Right. Mapping is done. Yes. That's okay. What I'm saying is the other part other than the mapping, like the conversion, for example, um in the source we have employee and we want ee right now in the new format it says self right so so looking at the column we know that this column b goes to c right b in source goes to c into standard but then we because it's a new format we know nothing about it we don't know if it says self and we don't know how to convert that into ee we know The code we have in ease. It can convert employee into EE, but we don't have the code to convert self into EE, right? So that part should also come from the AI. Either AI does it for us or it gives us the code to do it right. So the two scenarios understand so mapping is OK. Yeah, as long as we are OK with self or employee or EE then it's OK. All we need is the mapping. Here is the header. Us the mapping and we'll do the rest. We don't need to send you anything.

27:33 - Matthew Prisco 
But then there are other things as well. There's even another option where either we can give it the hard code that says if you see employee convert it to EE or we give it column by column the actual data that needs to be converted where we say this is row one and The data point so we don't give it the whole file, but we do each of those conversion steps and then make um You know without having it be hard-coded have AI give us You know

28:18 - Ashfaq Ali 
And result right For this employee mentioning himself, we call it EE. Now the AI would convert employee and self and person, it would convert that to EE. So we don't need the hard rules anymore if we can use AI.

28:33 - Matthew Prisco 
But what I want to avoid is giving it just the whole file and saying, do all of this. I think there's too much room for error and we can't check it again. We might be able to do it by field and make sure nothing got moved.

28:51 - Ashfaq Ali 
That's why we would ask AI to give us the code that we would run.

28:57 - Matthew Prisco 
Well, but those are two different ones. One is give us the values according to our general rules. The other would be, yes, give us code to plug in.

29:08 - Ashfaq Ali 
And here is the sample 10 lines, not the entirety line. 10 rows. Use that. Here is our desired output. Output like this, and give us the code that we would then run. But this would become a manual process. So each time you have a new format, you would do this workflow. And otherwise, you have a new file, you don't need it. You just give it and get there.

29:34 - Matthew Prisco 
I'm asking for one time use code. I mean, maybe to an engineer that doesn't sound scary to me, that sounds scary, that we need to have it code and then run that code. At that point, I would rather just have the AI do the values and just convert it, not ask for code. So that's my initial reaction, but I think we're thinking about this the same way. We need to look at their table maybe as the next step and say, what do we need to set up the master? However we have it now, is it going to be less let's convert that to an Airtable friendly version, or maybe there's some differences. We don't need certain things or we need extra things in order to do it from Airtable. So I think that's the next step is come up with our plan for that.

30:30 - Ashfaq Ali 
Do we need to identify what table that would be, what base that would be in?

30:37 - Matthew Prisco 
Like using Airtable as the megacensus data store? As the master. I think you guys call it.

30:44 - Ashfaq Ali 
Okay. Um, so that we can see what census files have we asked for?

30:50 - Matthew Prisco 
What format did it come in? What format did we change it to? What was the date?

30:57 - Ashfaq Ali 
Uh, what we could store the CSV, uh, be in Google sheets, the actual files with all these source intermediate sheets that would be in Google sheets or would that be It would be, right, so we would create a folder,

31:13 - Matthew Prisco 
still put the Google Sheets, if that's how we're storing it, we would store them there, but we would have it put the URL to the sheet, it would go to Airtable instead of go to the master sheet now.

31:27 - Ashfaq Ali 
It should, or you can have a sync between, even if you have the master, you should sync it with Airtable. So Airtable has all that information.

31:37 - Matthew Prisco 
That's definitely We should do yeah, so Identify what base and what table the mega census base? Do we want to have it our own base just for this we could we could start there we could always move it in with something else We that message you sent me the other day I think it's the reason

32:00 - Dwayne Gibson 
why to have its own base period is that when we organically grow a base like we are we're setting a timer for when we can Have to upgrade by customer success, right? That we're at a point where we're gonna have to like reevaluate that I think we need to find certain tables and get them out of there.

32:20 - Matthew Prisco 
Yeah Well, that's the same thing here is do we want to start from here's a clean base? We're not sure exactly where we want to put this on But if we found that it interacts some other table a lot, we would want them in the same base so that we don't need to do sync tables across bases. But I think at this point, we don't know what those bases would be. So let's start it in a mega census base, clean start, find the current master Google Sheet workbook and put whatever fields are on there. My guess is that we're gonna be able to add More fields that give us more control or more more context of the format that we got it from and so I said that because we already started a mega census

33:11 - Dwayne Gibson 
phase. Yeah, so I'm looking at we have an analysis table as well, so it looks like we Definitely got farther with the Google Sheet project, but we did put some thought into what we wanted to look like from this I'm

33:26 - Matthew Prisco 
guessing we have columns, but we don't have I could definitely make I'm saying just delete all the records if we were testing it or create new test records start from whatever columns that that we Had already added, but now

33:45 - Ashfaq Ali 
let's go from there Tell me this this idea of having any source file or any destination file is it like for you guys to like who would be the end user for that process? The employees of Metafix or your customers directly?

34:03 - Matthew Prisco 
Yeah. So just to give you an idea of what happens on, you know, when we're trying to use this, I would say, Dwayne, we need to get a certain quote and it's got to be in Benefiter. So we need to run, we need to get the group into Benefiter, which means we need the fastest way that we can get a file that we can upload immediately to Benefiter with no errors. But that's a common one where we'll do Benefiter as a part of the process. Then I'll say we also need to get a quote from this and this and this. Maybe one of those needs to be done electronically through their portal and they have their proprietary format. It needs to upload without error or we're going to get stuck. We don't want to be making those transformations manually for for sure. And then maybe a third category is we're going to email it as a part of a quote request, and they are not specific about what format. Whatever you give it to them in, they're going to work with it. So we can kind of say, I need it in this and this and this. Or then maybe we come back a week later and we say, oh, we need to do another quote. We've already gotten three versions of this sentence. Now we need to put in another request where it would use the same source file. It would use the same beta fits. We've already done that. Now I want to request an additional one. Or now here's where AI would come in. Here is a non-standard output file that we need. Let's have AI help us do a one-time mapping to that one. We're not going to do that now, Now I'm thinking about the inverse of what would be possible.

35:55 - Ashfaq Ali 
That is possible because in that scenario you know exactly the input that is the standard version. So the input is the standard version and we know exactly everything about it. What the column means, the format that it has like it would have EE. We know for sure that employee means EE and we we will not find employ yourself or anything else. That case, the AI thing could work. This is the non-standard output we want. This is the input, and this is the way it is. Just give us the output. That could work.

36:34 - Matthew Prisco 
But I think we also would be able to use this if we are creating a new template or a new output template. We say, hey, we're going to start using this new program and they have their format it needs to be. We give, we always have a CSV where they say they want you to upload it on this really Excel. We could take that and give it to a different prompt that says, give us the mapping, and then we can worry about the rules, but at least the mapping would be easy. So whatever process it took to do the output formats manually, we could have an LLM step to create new output formats. So now that we're thinking about the base, I think we wanna have some records of what are the input templates and what are the output templates. So if Dwayne, we already started an analysis table, I don't know what would have been on that. We would wanna have a list of what are the input and output templates. Maybe that's one table, it's just one field to say, is this an input or an output? And then is it operational, roadmap, requested, that kind of a thing. I'd be able to use that table to say, we've got this input and we want these outputs so that when we now run the new version of this, it would use those fields complete the workflow.

38:11 - Ashfaq Ali 
See, the way I have coded this mega census, we have in the master file a sheet called conversion rules. And that these conversions from male to M or employee to EE are not hardcoded in AppScript. These conversion rules are defined in this conversion rules sheet. So we can define new rules and the code would use that and convert the code. So we already have some sort of structure already here. We can give this conversion rules to AI. Right.

38:52 - Unidentified Speaker 
Yeah.

38:52 - Ashfaq Ali 
Okay. So give us the mapping. These are the rules. Convert that. Yes. We can over time add new conversion rules to it. Yeah. That was the idea behind this. This is great.

39:04 - Matthew Prisco 
And I think good timing because Trust Ali will be able to work through this. I have like 85% of the picture in my head that part I think Ali can can work on what are our options, but this would be relevant to an ROI workbook right now Isaac and Dwayne are working on Conversion rules literally we're using the same terminology Where we have inputs of the plan attributes the dental the medical dental and vision plan what are the deductible all of those things where it's the same thing we're talking about the census application of this but we've already built extraction automation so the source file instead of being a CSV that we've never seen is a PDF that we've never seen and it's going to extract and structure certain fields that we can expect on those documents some of them are the group variables what's the weighting Period so we're extracting. This is our you know process that we're trying to scale is just give us your documents We'll take everything valuable from there put it in our standard format so that we can put it into our Workflows into our deliverables into our Forms that you're seeing that the fill-out forms and they'll see oh wow It's already got all of our info because we've extracted it from there their benefit guide? What are their rates? What are their plans? We've got the machine learning algorithm to score the medical plan. Well, what do we need for that is we need to take whatever's in their benefit guide and start with that. How many plans do they have? Create that structure. Then we take the SBC, if it's medical plan or dental vision, we just call it plan summary. We need to take it from that carrier slight differences and put it into our standard format so that we can use it with the machine learning and so that our version is clean. So I think that we do mega census here and then I think Ali can follow over to the whatever we come up with for putting the conversion rules. How do we do that same solution for attributes, where does that live? So we were working on this the other day. We have the data dictionary, which we don't have up to date necessarily, but the concept is every field we need to have in our data dictionary. If there are conversion rules associated with that field, if we're talking about a census field, we would put the conversion rule in the conversion rule table so that we can always use it as needed. This would just be another version of that where rather than being census-related conversion rules, these are plan attribute conversion rules. And what are our options to have AI do these conversions? And what are our options to have them be more hard-coded but repeatable or flexible to where we can use them inside of other workflows? So I think that that's, That could work.

42:27 - Ashfaq Ali 
Maybe if AI can give us the mapping for an unknown source file. It can give us the mapping from an unknown source file to the standard version. And then we use these definable conversion rules on those columns.

42:50 - Unidentified Speaker 
Correct.

42:53 - Ashfaq Ali 
maybe we can come up with a desired output. Yeah.

42:58 - Matthew Prisco 
But the difference, the difference is we would need to give the whole file probably.

43:05 - Ashfaq Ali 
We are only asking for mapping, only the header. That's it.

43:11 - Unidentified Speaker 
Right.

43:11 - Matthew Prisco 
Well, but I wonder how could we, how could we give it enough info about the format for each of the fields to know which actual rules to apply. Can say, just give us the desired output. We don't need to give it every possible hard coding.

43:37 - Ashfaq Ali 
Or we can, like the current script reads these rules from this sheet and then apply it one by one. We can ask AI to create these rules and give that and then the script applies that one by one. Right. Instead of asking the code, ask him the rules, the conversion rules. Yeah. So we give him the headers, some rows, tell him from the header, give us the mapping. And from these three or five or 10 lines of sample rows, give us the rules, the conversion that our code would apply one by one and then we can hope to have an unknown input converted into strategy.

44:23 - Matthew Prisco 
Just by talking this through now we've identified if we were to have this not as a standalone base it'd be data strategy that either this would be in the same base or we'd have to have a lookup or a sync table from one to the other. So for example we would to have if we're using these conversion rules, we'd need to have a lookup to the conversion rule of the conversion rules would need to sync to mega census. Anywhere where we're using the conversion rules, we'd have to do a sync. Or is it possible or better at this stage to just throw them all in the data strategy and then not need to do any lookups? Because we're going to need that conversion rule which is already we were planning that it's going to be rather than have just fields, we could have for every data dictionary record, we could have a few fields that are conversion rules if they are used. So, is that a better data structure where we have these, you know, few columns that we're not using for most of the fields? Most of our fields don't have conversion rules. Would it be better to have a table that is only records of conversion rules and then we can tag them to whichever multiple fields might need those same rules? So if we make a rule, there might be two or three similar fields that use those same exact rules. That allows us to do that. What I'm saying here is that now if we have this conversion rules table, we can sync that in other bases. Wherever we're doing things like this, or does it mean that we should just try at the beginning to put this new mega census table, rather than be a mega census base for the time being, let's start it in the data strategy base and then we can bring it out later.

46:28 - Ashfaq Ali 
Where would your code be? Is it going to be in Python? Your data is in Airtable, where would be your script? Like right now, everything is in Google Sheet, the code and all of that. If you're new planning, if you're keeping data in Airtable and they're running the script that runs, calls the AI and all that is that in Python, then these rules should be part of Airtable.

46:58 - Unidentified Speaker 
Yeah.

46:58 - Matthew Prisco 
Well, the thing is, we're still trying to conceptualize What does the rule mean is it is it a Is it just something that we write is it something that we have? Like an example or So like we would need to tell it For a date of birth, what is the acceptable format and To do we just give it an example How do we do that if it is M male female? How do we do that? So we need to I don't have the answer to that yet.

47:35 - Ashfaq Ali 
We need to work through it See the rule that I the way I've defined is that I've said that in the column header that says sex the header is sex If the input is either he or M or M a n man Then the output that I need is M a l email Right this is the rule right this column if you see any of these options H-E-E-M and M-A-N, man. Right. Converted to male. Correct.

48:02 - Matthew Prisco 
So we may be able to have that same concept, but have it give more possibilities. So we could say, we could say if it says M, keep it as M. If it says male, it becomes M. If it becomes anything that starts with an M, becomes M. Anything that starts with an F, it becomes F because also the possibility that there's some human error even on the other file. So maybe the easier rule would be anything that starts with M or F.

48:37 - Ashfaq Ali 
Yeah, but we would ask AI to create this rule, this input that I have created manually that says either it's E or M or man converted to male. Would look at the source and then create the rule.

48:53 - Matthew Prisco 
If it says anything else right well I'm at the point where there's how would we need to store it so that it can be used by a formula versus how do we just put the rule so that we can

49:08 - Ashfaq Ali 
store it you want to store it because ai would give it to you for that particular input file you will use it that's it you want to store it anyway I read the file you got the desired output from AI, you used it in your script, that's it. Next time you have a new file, you give it back to it, it will give you new rules, you apply it and that's it. Like you use it and scrap it.

49:36 - Matthew Prisco 
There's no need to store it. I'm trying to think about how do we make this reusable across any time we're doing conversions.

49:44 - Ashfaq Ali 
So we can think about it for, for mega census, but I would want to store it. But then you have 10 rules. You got a new file, you added two more rules to that. Now you have 12. You cannot use those 12 because there could be another 13th rule in the new file. So every time you would have to go to the AI to get exact rules for that particular file. Even if you have 100 rules, you are not sure that there cannot be 101.

50:12 - Matthew Prisco 
I think we just need to try it in practice. So I think the next step is create the base? Dwayne, we just got to decide, do we put it in data strategy for now so that we don't need it?

50:26 - Dwayne Gibson 
I think that it doesn't, until we have a reason to put it in data strategy, looking at data strategy, we already have a mega census base.

50:35 - Matthew Prisco 
I think I'll give you it very quickly. The reason is we need to put the conversion rules in the conversion rules table of data strategy that we already started. We maybe haven't added any and we were thinking about that first in terms of plan attributes, but now we need to have, ideally, the same format that we store a conversion rule for a plan attribute, we should store a plan, a conversion rule for a mega census. There really wouldn't be a distinction based on the project. Ideally, we would come up with a standard format field. We don't have to.

51:15 - Dwayne Gibson 
We don't have a convergent rule table yet and data strategy.

51:19 - Matthew Prisco 
And well, we need, I thought we were talking about this already.

51:23 - Ashfaq Ali 
We have to have that is stored somewhere to have these rules stored somewhere. What benefit would you expect when you have them stored somewhere?

51:33 - Matthew Prisco 
Like, well, now we have a bad version of this in, maybe we should share this, Dwayne, if you could post the, um, Is it an ROI workbook? Where, where do we have the conversion rules that Anita has been working on for the plan attributes? I don't know where the file lives, but I can find the file. Yeah. Find the file. Let's give that to Ali. What I'm trying to do is to say, we've been trying to develop these rules from just having Anita add, add to this, uh, Google sheet and say, okay, well on here, we want to have, if it's the premium for for the so think about the difference between if it is a Deductible we want it to just say $1,000 with the dollar sign and with the comma But no decimal no to you know decimal places, but if that is insurance premium or the employee contribution amount now it needs to have a decimal and to Places so we would have a conversion rule that says we're trying to make deductible follow this rule nope now we're trying to do either Insurance premiums or contributions use this rule, so we've got this table already at this point It's only in Google Sheets But we need to come up with some way that the system of record has all of our conversion rules and like I was saying do we want to have each rule associated with a record on the data dictionary. And I think that the thing me and Dwayne were talking about last week was we need to have a table just for conversion rules so that we could apply the same rule to multiple similar fields that are on the data dictionary table. And in this case, so that we could sync the conversion rules table to an external Base or we need to put the base the table that we need in the same base so this rule that Anita writes is that

53:46 - Ashfaq Ali 
in English?

53:47 - Matthew Prisco 
I have no clue Right this is what we need to solve now.

53:53 - Ashfaq Ali 
I'm going to change some words now what you're calling a rule from Anita would become an instruction for AI and then AI would return technical code for that, which I have in my conversion. What I have is a technical code. Correct.

54:12 - Matthew Prisco 
So we would make that a field. The code, your part, would become a field on this table. So we have both. We have the English version, then we have the computer version or the code version. And what I'm saying is that this Only keep the English version because you're that either human or AI.

54:35 - Ashfaq Ali 
They both need English version.

54:37 - Matthew Prisco 
But what I'm saying is now we can take, Dwayne, hopefully you're able to find that file or we'll ask Anita.

54:46 - Dwayne Gibson 
I pinned it to the channel as well.

54:50 - Matthew Prisco 
So now we can give that to Ali so he can see this is now planning for after we get through mega census side as quickly as we can. I also want to do this mapping for conversion rules for the attributes because really one of the challenges is we've got multiple workflows that are extracting similar data. So we would take a certain plan attribute and it's coming off of a benefit guide. So we need to put this into the prompt and into N8N to take whatever we are getting and putting it into our format. But we might get that same attribute off of the plan summary, the SBC. If we change our rule or our standard, we now need to go back and fix it in multiple prompts. So what is the most scalable way to maintain our own rules, our own standards, and then we can think about ways that we could sync that or use that copy-paste a block of the rules that are relevant based on the fields that are in the prompt to have a little bit more scalability of how does it interact with NADAC. So this was already a pain point that we've been discussing. I think the insight here is that the same type of notation and record keeping for the plan attributes is the same way we want to try to structure the census related ones, because at the end of the day, they're all just data standards on fields that we encounter. We encounter them in different projects right now, but this is, I think, exciting that we can have a solution for this.

56:43 - Ashfaq Ali 
So what I can think of right now is we don't give AI the entire data. We give it the headers, ask the mappings, then give him these rules as prompts, as instructions, and ask him a piece of code in written that the standard script would run on that new file and would hope to have

57:15 - Unidentified Speaker 
it all converted, properly mapped.

57:18 - Ashfaq Ali 
And then, We can think of how we would do the calculations like right calculate the H Based on this data.

57:29 - Matthew Prisco 
I think I think that the steps though are let's take the the master panel and Rebuild it in air table. That's step one then then we evaluate the coded inputs input files options list them and table to say what are our what are our files that we've already done just we can even see it as a list is this an input or is this an output and then make it so that we can run mega census as we have it ease is the best example because we've already got an automation to pull an ease census so we would want it to be a smooth process to say this is an ease and we want it in these multiple output files or formats.

58:17 - Dwayne Gibson 
Are you describing a table that we upload a file to, we have a column that we have to drop down for?

58:26 - Matthew Prisco 
Yes, exactly. Which is the master. Right. So it's similar to the master, but now we're using it rather than just tracking it.

58:36 - Ashfaq Ali 
Instead of ease, you would say an unknown format. Right.

58:40 - Unidentified Speaker 
Yes.

58:41 - Matthew Prisco 
That would be one of the is trigger that workflow. Exactly.

58:45 - Dwayne Gibson 
How much do you think it would take since Airtable can use JavaScript to just copy paste our files over and adjust them? Will we need to change anything since the, well, I guess it'll be a CSV.

58:59 - Matthew Prisco 
Do we have any reason to send over the ones that we have? What do you mean the ones? What are you talking about?

59:08 - Dwayne Gibson 
Whatever's in the, the, the current. They're like 60 transformations. That we've done.

59:13 - Matthew Prisco 
Well, that's what I'm saying is, does it serve us anything to put those in there?

59:19 - Ashfaq Ali 
I would say, let's just start from today, capture the format and do the same thing, but we don't need the whole historical data. The script, the one that is going to orchestrate all of that, calling the AI and all that pre-built conversions, that script should not be in Airtable JavaScript? Or should it be in Python or something? What's your planning?

59:41 - Matthew Prisco 
about that. N8n is an option. N8n would be an option to say, where we're going to say, here's our code and N8n determines based on this variable, this variable, which function to call, which code to run.

59:57 - Ashfaq Ali 
N8n would not be a good choice for this conversion code. It would be a good choice for maybe calling the AI and all that. Yeah. Not like the conversion that we already have like e and all that just works and all that don't use any no code tool for that either use python right or so so for anything python right now it's either running

1:00:23 - Matthew Prisco 
locally which is probably what it's doing but for scalability all of this is not megasynthesis is running nothing in python it's all app script right it's all app script so so we need to say run this without, do we

1:00:39 - Unidentified Speaker 
use Apps Script at all?

1:00:41 - Matthew Prisco 
But if we're running it as an automation, even using Apps Script, now it's a Google whatever project thing, as I understand.

1:00:50 - Ashfaq Ali 
Instead of Apps Script, let's decide that we're not going to use Apps Script, but then don't use anything or any other tool for that. Use Python or Java programming language. No, I think it would be Python.

1:01:06 - Matthew Prisco 
and we would use Python anywhere for now to host it and run it that way, but put the URL back into Airtable. If it creates a Google Doc, but, or...

1:01:18 - Ashfaq Ali 
Yeah, yeah. The source, the master structure is in Airtable. The script that's going to run it is in Python. Python would read Airtable, call this script, whatever, type it back, and the output file would go to Google.

1:01:34 - Matthew Prisco 
going to try. But I do think that N8n can be involved in the routing of which function to call. So now we would create, I think what you're talking about is the Python script that would be run on Python anywhere. But N8n could say, this says unknown or use AI. And if we give it the file, that is an option. Airtable might even be an option to just run the automation and say, if this run, this run, blah, blah, blah. So we can decide, do we do we do that from Airtable automation? When we upload a file, look for, you know, once all of these five fields are complete, what is the source? What is the destination? Or even better would be have a button that says process, but it would need to have all of that logic of which functions to call. And we can decide, do we do that in Airtable we do that with NADEN? Let's assume Airtable for now.

1:02:34 - Dwayne Gibson 
Well, why would we even need NADEN? If we have a dropdown selector and we select input is ease, and then we select output is beta fix, we can pass both of those to the script wherever it lives. The script automatically knows what to do.

1:02:51 - Matthew Prisco 
It would start in Airtable, but one of the options would be rather than calling the Python anywhere, webhook to run the main conversion, if it's an AI one, it would go to, we would build the workflow with the LLM, we could build that into N8n. So it would say, we're using AI, the webhook it's gonna use is the N8n one, so that it can start, go through that workflow and then come back in to Airtable from there.

1:03:26 - Ashfaq Ali 
Yes, your idea is correct. But practically, even if it's an AI based workflow, it's still use an agent just to route things, not to process things. That's what I'm, my suggestion is based on my experience. For anything that's processing, create a Python script or JavaScript. But the question should not run the process, should route it.

1:03:55 - Matthew Prisco 
Airtable, JavaScript, or N8n.

1:03:57 - Ashfaq Ali 
Doesn't matter. From the work that we've been doing, where N8n is helpful is for the LLM step.

1:04:06 - Matthew Prisco 
So, you know, we could think about the audit, where you built a way to run the audit script using an LLM without it touching N8n. Do we have more controls for data input? Output by structuring the LLM side of that inside of an 8n as opposed to running directly to the the API for the LLM that's maybe where we're thinking about it differently is I'm saying do we need that level of customization or control on this prompt so that we're not going directly to the LLM.

1:04:49 - Ashfaq Ali 
I think that you would still need a language to do the processing even if it's calling the AI because AI you're providing something AI is going to give you something then you're going to work it's going to be a standardized journalized code right that journalized code is best suited for Python or JavaScript or programming language not a no-code, low-code tool like NHN. That's what I'm saying. Even if it's AI generated, that is standard code that's going to work with the help of AI, convert a new source into standard.

1:05:30 - Matthew Prisco 
I think we have a few steps before we get to that.

1:05:35 - Ashfaq Ali 
But we can look at that. While you are busy with the structure on Airtable and all that, maybe just to test this theory that we can we can actually use AI to do all this unknown conversions. Maybe let's test this with Apps Script because we already have the structure here.

1:05:54 - Matthew Prisco 
Or really, we need to write the prompt. Usually, the way that we test this is take the CSV. In this case, we need to make a sample that has all of the headers and then a few rows, 5, 10, whatever we want to test it with. It can be fake data. So that we can write the prompt and say, given this, we need an output of these fields that we need for the mapping and then tell us which column is that. So we need to devise the prompt and then test it first by uploading it directly before we bother with putting it through. But whether we use the API directly or whether we put any of this through any then we need to write the prompt and test The the accuracy is this feasible. I assume we can do this. This doesn't seem hard in terms of what an LLM can do that and Apps Script first script is my point is we just do it we come up with the prompt and And you do it manually before we do any automation We need to feel comfortable with the prompt and the input and the output and doing it manually by saying this is the file You're using chat gpt. I am uploading this file Edit the prompt I think first step ask chat gpt write a prompt to do this Then we start using that we iterate on the prompt and only once we are satisfied with the output results Then we think about do we do app script do we do n8n? But before we we cross that bridge we need to First do all this other work for the panel the master, what does that look like in Airtable? Then I think we need to do a little bit of work of how do we route the ones that are hard-coded, make sure that will work. Then we start experimenting with the prompt itself. Then we think about is it going to be Apps Script or Airtable or N8n to run the... Okay, there's something big we need to discuss.

1:08:00 - Dwayne Gibson 
We tabled this, we pushed it down the road the first time with mega senses and it's come up the output beta fits format the last time what we're currently calling intermediate yes yeah intermediate equals beta fits format we took that from a random rippling file yeah and it's decent for medical but it's options for dental and vision and anything past that leaves a lot to be desired right which is why We need to have each field where we're

1:08:34 - Matthew Prisco 
saying this is our rule so that we can see it easier And we can edit it, and if we said yeah, we we hit this down the curb But now we need to pay closer attention just to this field I want to be able to focus on that field from our conversion rules table rather than from However, it's buried in the way.

1:08:57 - Ashfaq Ali 
We have it now Is there some? That you can give to AI so that it can understand the terminologies of your business. So that it can, when it reads the header, it knows what it means like a deduction or contribution and what it means. So, so it's easier for AI to make sense of those headers and we need the mapping.

1:09:19 - Matthew Prisco 
This is data dictionary. We need to say, what are we going to call it?

1:09:24 - Ashfaq Ali 
So we could, some standard document.

1:09:26 - Matthew Prisco 
Yeah, we don't have that, but We what I'm saying is that we would need to already make decisions or start making decisions. Do we call it deductions? Do we call it blah deduction this this like, If you don't call it maybe but the source that you get the random source I'm saying that we are calling it something in air table and what should we call it and we could give that list and say this

1:09:51 - Ashfaq Ali 
is this is we don't even have our field names where part where you tell AI desired output, this is what we call it, this is what we call it, but the input, you don't have control over it.

1:10:04 - Matthew Prisco 
We're hoping that we have some consistency. There's what are we calling it on ROI workbook, needs to be the same thing we're calling it in Airtable, which may or may not be the case. So I'm saying that we need, we are, because we don't have a data person who's just focused on what are we calling these fields and why, we are hoping that we have some, you know, consistency between Airtable and ROI workbook and anywhere else. We need to have, this is our field so that we could say we're going to use this same terminology and put it in a prompt to help us do this. But I'm saying that if we took, we have two different data dictionary tables in two different bases, three different ones, two that are totally different CRM based versus customer success. We want to put those into the data strategy base so that we eliminate the redundancy in those fields. But we need to go through that process and try to also incorporate it into this new one. So I think, Dwayne, this makes me upgrade the urgency to find a data dictionary experienced person who can work on what do we want to call the fields. So that we can use it in the rest of this. But I don't think it will get in our way.

1:11:29 - Ashfaq Ali 
It just, it's messier. I think since you have a paid version of ChatGPT that you're going to use, I think it can like store your preferences or your history or your, the way you like things.

1:11:44 - Dwayne Gibson 
I've been experimenting or making a custom GPT. Yeah, this is good. Do that. And it can be very quick, right? So we don't have to make it perfect on the first try. We can download the three versions of the data dictionary, give it to a custom, let's call it the Betafix data GPC.

1:12:04 - Matthew Prisco 
And say, give us one version of this. But that's what I'm saying is that the source of truth needs to be the data dictionary in the data strategy base so that we can use that in a custom GPT, so that if we say this is a resource that we have, it has that already. We don't need to make a Google Doc that says, here is our dictionary, or here is our industry, blah, blah, blah. We would use the actual source data so that if we make any changes to our data dictionary fields or our conversion rules, we can make those resources in other GPTs, other rags, anything like that. But Dwayne, if you can do that, please try to...

1:12:49 - Dwayne Gibson 
Matthew, so I think you should make the custom GPT and then just share it, and then we can upload the files because...

1:12:57 - Matthew Prisco 
I don't know how to make custom GPTs, but you already said it. Take those three files and...

1:13:03 - Dwayne Gibson 
That's what I'm saying. If I make it with my paid account, I'm not sure I'm going to continue paying for it. So if it becomes something I don't need next month, our GPT is gone.

1:13:14 - Matthew Prisco 
We need to have a beta fits Account that anybody can log into and do if we're gonna do custom GPT's, but I think regardless of that We need to can Consolidate the three tables and say give us one table and give us some best practices and structure Which I was gonna hire somebody to do that and and put that right back into see if Chad GPT can help us to do the consolidation of all of this so that then when we're doing the conversion rules, we're linking it directly to the official version of those fields. And I think we still should get somebody just to, you know, I'm going to put data dictionary in air in Upwork and see see who's done those kinds of projects.

1:14:08 - Ashfaq Ali 
So the prompt is the first step. Right he would write the prompt pick a random format And give it the intermediate format the fit of it right and ask him to do We would be able to give it our list of fields which we could generate

1:14:26 - Matthew Prisco 
from the data dictionary I think we would need to tell it or we would need to give it. Here's here's our Here are the list of the fields that are on our Template that are on the the standard intermediate. We can just give it that Yeah, just give it the head Right, but what I'm saying is that the names of those fields are subject to change as we go through this process But giving it something and saying we need it this this and this here's the rule associated with each of those those would now be two Columns that we can paste into the prompt so that it has it and if we need extra context a third column, a fourth column. Here as a part of the prompt, now we have something that we can iterate and replace it as we get more advanced with, if we make any changes to the name of the field or the conversion rule or anything else, it becomes a component as opposed to just we've typed it into a prompt. So who's going to do this first step, the prompt? Wayne will will work on the Consolidating the Data dictionary fields you can also have autonomy on the the rules what fields do we need what? Notation do we need to use are you able when you're an air table to create new fields? We have the same login as all of us what I'm saying is is as you see fit We need to create the conversion rules table and start taking the ones that we've already done for the other attributes and figure out how do we get it from the Google sheet into a scalable format in The conversion rules in air table then we need to do the same thing that we've already started Attempting for the plan attributes and do it for the mega census related rules, too Link them to the fields that Dwayne's going to work on Consolidating these these different tables, so we have a source of truth, and then I'll find somebody to really own that moving forward

1:16:49 - Ashfaq Ali 
okay, so Do I have the chat GPT account credentials?

1:16:53 - Matthew Prisco 
I don't remember exactly any all we're doing is writing a prompt so whatever the prompt is as we're iterating, we can share it, say this is the prompt and you can use your chat GPT account to get it to write the prompt. And then when we're ready to scale this, then we put it into N8N or into code.

1:17:16 - Dwayne Gibson 
I'll share, I'll share a GPT with you, Matthew. The difference between the free tier GPT and the paid ones is massive now in terms of the logic and how well it works.

1:17:27 - Matthew Prisco 
So not having access to 4.0 And 4.5 is a pretty big But but I don't think I don't think that the first version of the the prompt needs a GPT It's just here's our list of fields give us whatever it could be Done with the GPT with here's the here's the rules and here's whatever and you tell it consult this So is there a free account on chat GPT?

1:17:55 - Ashfaq Ali 
that I can create and use? You don't have a way to use ChagGDT? No, I don't have. I've just recently started using DeepSeq only. Okay. But you could use DeepSeq too.

1:18:06 - Matthew Prisco 
So again, all we're trying to do is draft a prompt that we could use with whatever LLM we want. The prompt ends up being the same. Right now we have no draft of the prompt. We need to get something that we're comfortable with that could be used any LLM of our choosing, and that can be created as a part of an N8N automation, we would say, this is our prompt. So even in our project management system, we have a prompt table so that we can say, this is a record, what's the purpose of this prompt? What's the text of the prompt? So we would be able to edit and version control the prompt. So the prompt becomes the asset. You can use Deep Seek or any any LLM, and then you use DeepSeek as examples. I was thinking you had chat GPT, use DeepSeek. First ask DeepSeek, I'm trying to create this. How can I solve this problem? Maybe it has more to offer than we've already done. Then I would say, I need to create the prompt for us to do this. What prompt do you suggest? Then we edit that to have the actual fields. Then we start and we say, here's the test file, the one with the headers, randomly generated, can it do the mapping? When we get good results from doing it with DeepSeek, then we think about, do we wanna do Apps Script or do we wanna do N8M? But we need to develop the prompt and make sure that it works directly with DeepSeek.

1:19:44 - Ashfaq Ali 
Okay, so let's start something and see how it goes, and then I'll share the results. I think that's good.

1:19:51 - Matthew Prisco 
Okay. Okay.

1:19:52 - Ashfaq Ali 
I'll start with the mapping first, then with the conversion rules, and then with the calculations. Good. All right. Well, I'm going to jump off.

1:20:01 - Matthew Prisco 
Ali, I hope it was a good Ramadan. I think we gave you a little bit of time.

1:20:07 - Ashfaq Ali 
We've been working on some other things, but I think this is going to be a fun- I really appreciate you holding few weeks just because I was busy. Yeah. But hopefully this is good because it accomplishes the next version of of mega census.

1:20:24 - Matthew Prisco 
But I think this gives me a clearer understanding of another problem that we were having on the scalability of the conversion rules for the plan attributes. So I think one will go very well into the other. And now I see who we can plug in. Focus on the data dictionary too. Dwayne had some great ideas here with how to leverage what we already have. So I think he can play with that. Ali, you can work on the prompt and thinking about how can we still use some of what you've already coded or how can we make some modifications to fit into this new workflow?

1:21:07 - Unidentified Speaker 
Okay.

1:21:07 - Matthew Prisco 
All right. Let's end it there. Thanks guys.

1:21:12 - Dwayne Gibson 
Ali, I'm sending you, I'm sending you, I made a chat GPT, allows you on the page here. It allows you to create a custom GPT. And what that does is, here, I'll show you right quick. What that does is it, you can upload files that become part of its knowledge base. So when you ask a question, it will have those files to refer to. So I gave it the ROI default template. I explained a little bit about it, the default template. Then I gave it a copy of every script and explained a little bit about what we were doing with each script. I haven't had a chance to really test it, test this knowledge yet, but I think what Because with the custom GPT, you get 20 files that you can upload. And so far, I think I only have I put all of the scripts into one one document, so they only count as one file.

1:22:25 - Ashfaq Ali 
And then the second file, I believe, is the template. The file that you uploaded. In this?

1:22:33 - Dwayne Gibson 
Here, I'll show you. So, we'll go to edit GPT.

1:22:39 - Ashfaq Ali 
So, if possible, just share me the chat GPT credentials because DeepSeek is also free and would have limitations on how much I can use. I don't know. So, I keep using DeepSeek, but if I hit a limitation, then I would need the credential for ChatGPT from you.

1:23:04 - Dwayne Gibson 
I'm going to send I'm going to send you this one now. That's what I was typing over here. So I'll send this copy to you. One thing I do. So I've been building a lot of just random applications using the straight code from using Google Gemini, ChatGPT, and I'm going to start testing Cloud And I've been building working web apps, right? Like I'll just get an idea and think like, okay, with one hour and strictly copy and paste, what can I build? I've learned a lot. One thing I've learned is that chat GPT is not the best for code because most of its knowledge base is outdated for functions and documents. So something that should take 20 minutes takes two hours Updating the code to current, but I was searching this morning and apparently anybody who's anybody already knew that and chat GPT writes bad code and Apparently Claude is the one you want to go to if you want flawless click code Well, I think the takeaway from all of this is that we can work a lot smoother and a lot faster if we have these custom GPT is like, can you see the screen right now? Yeah. So let's test it right now. Our ROGP, our ROI GPT. What purpose So this is the master script So the files I gave it were looks like I gave it two copies of the dynamic template accidentally and then one copy of the ROI scripts The master script, right? Yeah, so the file will look I don't have You won't be able to see that in my files, but I just, I just created a Google doc and then I went into the app script and copy and pasted every script. I gave it a header so it could tell where the new function started. And I just pasted the name script, name script, name script. Let's see if they can access them. So it gives us, it gives us a, pretty good description of what it meant to do. So, my goal with this is to be able to like have something that we can ask holistically about the entire project and say like this isn't working or we want to add this new function. What's the quick quickest way will this work to have an assistant basically that already knows exactly about the project.

1:26:22 - Ashfaq Ali 
So once you have the files uploaded would keep it that in history.

1:26:29 - Unidentified Speaker 
Yep. Okay.

1:26:30 - Dwayne Gibson 
So I'm thinking about turning this. But you can see like it did not do too great here because it didn't name Oh yeah. Five six.

1:26:48 - Unidentified Speaker 
Yeah. There's more than six. Yeah. There it goes.

1:26:59 - Dwayne Gibson 
Another thing that I've had to do is when you create a GPT it allows you like here I tell it to use the bluff principle bottom line up front because when you're working a lot with the LLM it just gives you a lot of extra wording that you don't really need like if you're gonna have if you're gonna be working for two hours with it you don't want to have to read eight paragraphs of explanation every time you interact with it. So I designed it to just be short, direct and give me the point right up front. Right. Don't. Don't write a book every single time.

1:27:50 - Ashfaq Ali 
So that's why it's providing you with the bullet points for each of these services. Yeah.

1:27:56 - Dwayne Gibson 
So I I've been and this is something I've been trying really hard to design it on. I wanted to ask me at the end of every prompt, do you want more details? If I don't ask for details, don't give me details. Just give me the data and then give me the opportunity to ask for more data instead of assuming that I want a long explanation. See, let me know if you need further explanations or more details on any of them. So you can do that here in instructions.

1:28:30 - Unidentified Speaker 
OK. All right.

1:28:33 - Dwayne Gibson 
You can also do conversation starters, which are not really that useful. The next powerful thing that's not really working the best is we have actions. So, we can call web services here.

1:28:55 - Ashfaq Ali 
So, we can use different authentication types. Yeah, we can try.

1:29:02 - Dwayne Gibson 
So every time it won't be able to access it without without you without without prompting you. So I built a rag and I set up a embedded pine cone database. And I want I uploaded it with some research articles and at first I was thinking I wanted this to be the same as like having it in his files. To automatically know, but that's not how it works. Instead, when it thinks that the data it needs to answer you is in an external database, it asks you, yes or no, do you want to allow me to access this database, which is just one extra step. And then you say, yes, you can access it. One of the problems with it is that it times out really quick too, but that's because I'm using the free service of all of these external API builders. So it does, the web service goes to sleep if you haven't used it in, you know, 30 minutes and it takes like 20 seconds to wake up and that's the timeout limit for cat. So you gotta know that you're gonna use the service, wake it up, then you can ask it, but they're working on it. It's definitely getting better. So that's another thing that we can do. Like we can expand the built-in data that uh it has but uploading files we only have 20 files here that we can upload but they can be pretty big so basically you are already in the process of creating a custom gpt for the ry project itself yeah so what I'm thinking and let me get let me get your opinion on this so I don't know how much more or I could give it just to make it ROI based. Like it's got all the files that are related to ROI. Now it has the, I guess I could give it the ROI master file. It has the scripts, it has the- But it doesn't have that much data.

1:31:11 - Ashfaq Ali 
The script is important for the master. So it knows what we are doing, how we are doing it.

1:31:18 - Dwayne Gibson 
Yeah. That's what I'm thinking. I'm thinking instead just to change the name of this GPT and call it BetaFITS GPT-1.

1:31:29 - Ashfaq Ali 
Exactly, yeah. My next question would be like, is it just ROI or is it like BetaFITS?

1:31:38 - Dwayne Gibson 
How about we call it BetaGPT?

1:31:41 - Unidentified Speaker 
Sounds cooler.

1:31:42 - Dwayne Gibson 
And since it's a second, version.

1:31:46 - Ashfaq Ali 
What should we do? 1.1? Yeah, why not? Well, ideal scenario would be to have your Betafix to have its own AI.

1:32:06 - Unidentified Speaker 
Agreed.

1:32:07 - Ashfaq Ali 
So, it's trained for your business domain. And you're not worried about sharing it with anyone else. Ideal, the ideal situation. So you just really give it any file of any of your customer and you know that it's not going to share it with any other company or third party. It will stay part of VitaFix itself.

1:32:36 - Unidentified Speaker 
No, I 100% agree, Ali.

1:32:38 - Dwayne Gibson 
Let me refresh this and share this link with you. So I'm going to share this with you. I am not going to share it with a lot of other people. So if you look here, I think the default, I'm not sure if you can change the model. So I think that the model that this uses is 4.0 itself. Go to chat GPT once you have a paid version you see 4.0 that is the current most advanced release okay but it um even the paid accounts have limits so I hit my limit on 4.0 yesterday because I was working with it probably three hours straight we were coding I was giving get massive files and it timed And But it's not even my favorite model. They have an experimental research preview 4.5 and it is. I would say if 4.5 is a high schooler, it makes 4.0 look like a third grader.

1:33:49 - Ashfaq Ali 
It's that much better. Try using. The old one model. See there are two types. I don't know much about AI, There are two type of models as of right now. One is like straight, you ask a question, it would give you the answer. That's it, right? Like a 4 and 4.5. But then there's one, it says a reasoning model. A reasoning model like O1, chat GPT O1 is a reasoning model. Similarly, DeepSeek has R1, it's reasoning and X1 is like sharing. Chat version. V1 and R1. So the reasoning model does not give you a straight away answer. It works like a human. If you ask him a question, a prompt, it would then create a task list, do online research, learn from its mistake, and then it will give you screen by screen what it's actually doing behind and why it gave you the result that it gave you with a reason and that is a lot more useful for what we are trying to do so you won't see just the output you would actually see how it's what it's working on how it's working on like if it's going on betafits.com and learning some keywords, or is it Googling what contribution means, it would show you that I'm going on Google and searching for contribution. Oh, okay, that's what it means, and then it's going to update its task list, right? So the first step, it would create a task list that get the meaning of all these key technical jargons, and then convert them into this mapping. Okay, I see this, and I see this, and that's why I did this, and all that. It's a reasoning model that's more suitable to you. So if you are going to pay more, don't go to 4.5, go to 0.1. Try that. Okay.

1:36:07 - Unidentified Speaker 
Yeah.

1:36:07 - Ashfaq Ali 
I don't know about it, but that's a reasoning model. Yeah. And then there's a new breed, a desired future version. It's called generative, no, journalized AI, something like that. And AI model that uses specialized models. Have you heard of Manos? Manos? No, I haven't.

1:36:33 - Dwayne Gibson 
But I've been to a hugging face. It's where all of the open source models are hosted, kind of like GitHub. But for API, for AI, LLM models and I definitely think like your idea of beta fits building one. The only thing people here are doing is building Custom GPT or custom LLMs for their business and whatnot. So with step-by-step instructions, too. So what models are good for which like Deepseek is represented a lot here.

1:37:15 - Ashfaq Ali 
Like here's a v3 model of easy check check version DeepSeek R1 is the reasoning model. And it's really outstanding from what I hear from

1:37:27 - Dwayne Gibson 
Here's R1.

1:37:28 - Unidentified Speaker 
R1.

1:37:29 - Ashfaq Ali 
Look, also look at for MANUS Manos. It's another Chinese company. Just have a look what it is. It's like the Holy Grail of AI kind of generalized AI model. What's the name of the company again? M-A-N-U-S. It's the name of the model, I think. And it's not public yet. No, no, no.

1:38:09 - Unidentified Speaker 
No, this is not.

1:38:12 - Ashfaq Ali 
M-A-N-U-S.

1:38:24 - Ashfaq Ali 
No, no, M-A-N-U-S, M.

1:38:26 - Unidentified Speaker 
M? Yeah. I see.

1:38:28 - Dwayne Gibson 
Oh, it still has invitation codes.

1:38:31 - Ashfaq Ali 
Yeah, just read what, how is different from a normal chat model or a reasoning model. General AI agent, that's what, with my limited knowledge, I think is the holy grail for everyone who is striving to achieve this. JetGPT also has some new model that may be a competitor to this.

1:38:58 - Dwayne Gibson 
I'm going to give you the wait list now.

1:39:02 - Ashfaq Ali 
So the kind of things that you are trying to do, this is the ideal solution, like coding and building apps and all that. R1 model, O1 model, DeepSeek R1, JetGPT O1. And Manos and all that. Maybe just have a look at the few videos on YouTube related to that, and you'll just get an idea of how many kind of AIs are there. It's a different kind of AI, like reasoning model a different kind of AI?

1:39:52 - Unidentified Speaker 
Yeah, I'll check that out now. Okay. Stop presenting.

1:40:06 - Dwayne Gibson 
Okay, let me, upload these files to beta GPT and I'm going to also take your suggestion and try to make a general document about beta fits the company and upload that as well so it'll have even more broad context

1:40:28 - Ashfaq Ali 
about beta fits and just another any your industry related you would find a lot of I think documents or websites or something just just so that your custom model understands technical words of your universe so it would help it

1:40:49 - Dwayne Gibson 
map things even better agreed all right so I shared I shared the conversion rules and I pinned it to the code team channel and I direct messaged you the All right, I haven't received a direct message yet.

1:41:20 - Unidentified Speaker 
No, hold on. Okay, got it.

1:41:32 - Dwayne Gibson 
Let me change that description. I would love it if it allowed people I share it with to configure it as well. I don't think it'll, you can try to see if it'll let you configure it, but I don't think it will. I think it'll just but you search it.

1:42:04 - Ashfaq Ali 
And just use it. Yeah. OK. So I'll start with what we already have, like give it JustWorks input file and ask him the mapping for our intermediate file and see what percentage output And also tested so that the way that the persistent memory has been

1:42:34 - Dwayne Gibson 
evolving. When I first started with the pro model, it did not have persistent memory from one conversation to the next. Each conversation would terminate all, you know, storage. But I think that they've upgraded it and it does create memories. So when you upload what you upload the file, you may be able to tell it to store certain amount of data in itself as the model itself for future conversations. So I think we find that out and it'll be helpful to us a lot.

1:43:08 - Ashfaq Ali 
Let's say if there's a mistake in one of the mappings and I instructed that this is the right mapping, it would then remember it and maybe in future would do better. Okay. So I'll just use JSWorks as the input and like what we already have achieved and see the accuracy of it. Get the conversion.

1:43:30 - Dwayne Gibson 
Use ease as input. I think we haven't got we haven't gotten many just works recently. So and we I think in the last 20 convergence I've done 18 ease. So it'll be a lot more important to do that.

1:43:48 - Unidentified Speaker 
OK.

1:43:48 - Dwayne Gibson 
Let me know if you have any questions Ali.

1:43:53 - Ashfaq Ali 
Okay. I'll just try it. Um, it might take some more, a little bit more time the next one week. Uh, but I'll definitely start working on it as much as I can.

1:44:08 - Dwayne Gibson 
Got it. Perfect. Thank you. Okay.

1:44:11 - Ashfaq Ali 
Thank you. Thank you. See you.
    """

    # Run semantic matching with improved parameters
    results = semantic_matching(
        records=records,
        query_data=transcript,
        top_k=10,  # Get more results for better coverage
        score_threshold=0.0,  # No minimum threshold to see all scores
        filter_current=False,  # Set to True to only get active PCFs
    )

    print("\n📊 Final Results Summary:")
    print(f"   Found {results['total_found']} relevant PCFs")

    # Display store statistics
    stats = get_store_stats()
    print("\n📈 Vector Store Statistics:")
    print(f"   Total PCF records: {stats['total_pcf_records']}")
    print(f"   Unique PCF records: {stats['unique_pcf_records']}")
    print(f"   Note: {stats['note']}")
