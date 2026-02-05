from typing import List, Tuple, Dict, Any
from graphiti_client import graphiti, initialize_indicies_and_constraints
from langchain_core.tools import StructuredTool
import asyncio
from pydantic import BaseModel, Field
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
import logging

logger = logging.getLogger(__name__)


class Search(BaseModel):
    query: str = Field(description="The query to search in the Knowledge Graph")
    top_k: int = Field(description="total number of results to fetch from the search")


async def search_graphiti_nodes(query: str, top_k: int) -> dict:
    """
    Search graphiti and organize results by project and component.
    Args:
        query: Search query string.
        top_k: limit the number of results.

    """
    try:
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = top_k
        node_search_results = await graphiti._search(
            query=query,
            config=node_search_config,
        )

        return node_search_results
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}")
        return {}
    # finally:
    #     await graphiti.close()
    #     print("\nConnection closed")


async def search_graphiti_edges(query: str, top_k: int) -> dict:
    """
    Search graphiti and organize results by project and component.
    Args:
        query: Search query string.
        top_k: limit the number of results.

    """
    try:
        node_search_results = await graphiti.search(
            query=query,
            num_results=top_k,
        )
        await graphiti.close()
        print("\nConnection closed")

        return node_search_results
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}")
        return {}
    # finally:
    #     await graphiti.close()
    #     print("\nConnection closed")


async def search_nodes_and_edges(query: str, top_k: int):
    try:
        results_nodes = await search_from_graphiti_nodes_tool.ainvoke(
            {"query": query, "top_k": top_k}
        )

        await asyncio.sleep(0.3)
        results_edges = await search_from_graphiti_edges_tool.ainvoke(
            {"query": query, "top_k": top_k}
        )
        return {"result_of_nodes": results_nodes, "result_of_edges": results_edges}
    except Exception as e:
        print("❌ Error occurred:", e)
        return {}


def print_edges(results: List[Tuple[Any, float]]) -> None:
    """Print edge results - results is a list of tuples (edge, score)"""
    if not results:
        print("No relationships found.")
        return

    print(f"\n🔗 Found {len(results)} relationships:\n")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        # Unpack the tuple - result is (edge, score)
        if isinstance(result, tuple):
            edge, score = result
        else:
            edge = result
            score = None

        print(f"\n{i}. Relationship:")
        print(f"   UUID: {edge.uuid}")
        print(f"   Fact: {edge.fact}")

        if score is not None:
            print(f"   Relevance Score: {score:.4f}")

        # Print edge type/name if available
        if hasattr(edge, "name") and edge.name:
            print(f"   Type: {edge.name}")

        # Print source and target
        if hasattr(edge, "source_node_uuid"):
            print(f"   Source: {edge.source_node_uuid}")
        if hasattr(edge, "target_node_uuid"):
            print(f"   Target: {edge.target_node_uuid}")

        # Print temporal info
        if hasattr(edge, "valid_at") and edge.valid_at:
            print(f"   Valid from: {edge.valid_at}")
        if hasattr(edge, "invalid_at") and edge.invalid_at:
            print(f"   Valid until: {edge.invalid_at}")

        # Print custom attributes if any
        if hasattr(edge, "attributes") and edge.attributes:
            print("Attributes:")
            for key, value in edge.attributes.items():
                print(f"      • {key}: {value}")

        print("-" * 80)


def pretty_print_result(result: Dict[str, Any], limit: int = 1000) -> None:
    """
    Helper for pretty printing the result.
    Args:
        result: returned dictionary from search
        limit: number of results to print

    """

    print("Top Entity Nodes (Name → Score → Summary)")
    print("-" * 41)

    for i, (node, score) in enumerate(
        zip(result.nodes, result.node_reranker_scores), start=1
    ):
        print(f"\n{i}. {node.name} (Score: {score:.3f})")
        print(f"   → {node.summary}")

        if i >= limit:
            print(f"\n... (Total Nodes: {len(result.nodes)})")
            break


search_from_graphiti_nodes_tool = StructuredTool.from_function(
    func=search_graphiti_nodes,
    name="search_graphiti",
    description="This tool is used to serch content in the Knowledge Graph",
    return_direct=True,
    args_schema=Search,
    coroutine=search_graphiti_nodes,
)


search_from_graphiti_edges_tool = StructuredTool.from_function(
    func=search_graphiti_edges,
    name="search_graphiti",
    description="This tool is used to search content using edges in the Knowledge Graph",
    return_direct=True,
    args_schema=Search,
    coroutine=search_graphiti_edges,
)

if __name__ == "__main__":

    async def main():
        # Initialize indices and constraints first
        print("Initializing Neo4j indices and constraints...")
        await initialize_indicies_and_constraints()
        print("Indices initialized successfully!")
        # search
        results = await search_from_graphiti_edges_tool.ainvoke(
            {
                "query": """
E-Commerce Platform – Quarterly Ops Review
Fri, June 21, 2025

0:05 – Matthew (PM)
Morning everyone. Before we start the new sprint, I wanted to do a quick retrospective on the E-Commerce Platform rollout. There’s been chatter about performance dips, particularly around checkout and inventory sync.  

0:21 – Lisa (DevOps)
Yeah, the Payment Service instances spiked CPU usage last weekend. Stripe and PayPal both had higher than usual retries. Might be just volume, but some of the logs suggest slow database commits.  

0:37 – Matthew
Okay, so that’s backend. What about front-end latency?  

0:40 – Unidentified Speaker
Search seems fine, but product recommendations are lagging — maybe related to Elastic or Redis cache invalidation.  

0:49 – Lisa
Actually no, that’s the Recommendation Engine. Its retraining job overlapped with index refresh. It queued about 70,000 product updates at once.  

1:00 – Matthew
That’s a lot. Should we decouple indexing from recommendation updates?  

1:05 – Lisa
Yeah, eventually. The bigger problem is, all microservices still share a single Redis instance. We might need to shard it by component.  

1:14 – Matthew
Good point. And notifications? Are SMS and emails still duplicating?  

1:19 – Unidentified Speaker
Yes, but that’s not a bug — it’s both the Notification Service and the CRM workflow triggering the same event. We can fix it with a single message bus, maybe Kafka, instead of direct webhooks.  

1:33 – Matthew
Okay, add that to the backlog. Anything else from QA?  

1:37 – Priya (QA)
Yeah, I’ve noticed inconsistencies between warehouse stock and product availability. The Stock Sync job sometimes reports outdated counts even though Inventory Service shows correct numbers.  

1:49 – Matthew
Could that be caching again?  

1:52 – Priya
Possibly. Or two parallel cron jobs overwriting each other.  

1:57 – Lisa
Actually I saw overlapping CRON expressions last week — one at `*/15 * * * *` and another at `0,15,30,45 * * * *`.  

2:07 – Matthew
Okay, let’s clean that up.  

2:09 – (pause)

2:11 – Unidentified Speaker
By the way, marketing wants to run A/B tests on the search filters next month. They need API access to the Search Service.  

2:19 – Lisa
Sure, but that could add load spikes. We’ll need to throttle.  

2:24 – Matthew
Alright, let’s keep that in mind.  

2:26 – (short pause)

2:28 – Unidentified Speaker
Oh, and one unrelated thing — IT reported that the office printer shows “Stripe Integration Timeout” again.  

2:33 – (laughter)

2:35 – Matthew
(laughs) Yeah, let’s not open that can of worms again.  

2:38 – Lisa
I swear, that name will haunt us.  

2:41 – Matthew
Anyway, good job keeping uptime above 99%. We’ll summarize action items under “E-Commerce Platform” in Airtable, not individual components.  

2:52 – Priya
Perfect. I’ll update those records manually for now.  

    """,
                "top_k": 10,
            }
        )
        print_edges(results)

    asyncio.run(main())
