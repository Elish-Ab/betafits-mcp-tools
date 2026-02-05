from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.llm_client.openai_client import LLMConfig, OpenAIClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
import os

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
URI = os.getenv("NEO4J_URI")
PASSWORD = os.getenv("NEO4J_PASSWORD")
USER = os.getenv("NEO4J_USER")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# Initialize Graphiti with Gemini clients
graphiti = Graphiti(
    uri=URI,
    user=USER,
    password=PASSWORD,
    llm_client=GeminiClient(
        config=LLMConfig(api_key=GEMINI_API_KEY, model="gemini-2.0-flash")
    ),
    #llm_client=OpenAIClient(
    #    config=LLMConfig(api_key=OPEN_ROUTER_API_KEY, model="openai/gpt-4o-mini", base_url="https://openrouter.ai/api/v1")
    #),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=GEMINI_API_KEY, embedding_model="text-embedding-004"
        )
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(api_key=GEMINI_API_KEY, model="gemini-2.5-flash-lite")
    ),
)


async def initialize_indicies_and_constraints():
    await graphiti.build_indices_and_constraints()
    # Attempt to create missing fulltext index used by querying code if absent
    try:
        if hasattr(graphiti, "run_query"):
            # Check if index exists
            existing = await graphiti.run_query("CALL db.indexes()")
            names = {row.get("name") for row in existing if isinstance(row, dict) and row.get("name")}
            if "node_name_and_summary" not in names:
                await graphiti.run_query(
                    "CALL db.index.fulltext.createNodeIndex('node_name_and_summary',['Episodic'],['name','summary'])"
                )
        else:
            # Fallback: try direct Cypher via internal driver/session if exposed
            driver = getattr(graphiti, "_driver", None)
            if driver:
                async with driver.session() as session:
                    res = await session.run("CALL db.indexes()")
                    records = [r.data() for r in await res.consume().records] if hasattr(res, "consume") else []
                    names = {row.get("name") for row in records if row.get("name")}
                    if "node_name_and_summary" not in names:
                        await session.run(
                            "CALL db.index.fulltext.createNodeIndex('node_name_and_summary',['Episodic'],['name','summary'])"
                        )
    except Exception as e:
        # Swallow errors; index creation is best-effort.
        import logging
        logging.getLogger(__name__).debug(f"Fulltext index setup skipped/failed: {e}")