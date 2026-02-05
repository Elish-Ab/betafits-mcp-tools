import logging
import asyncio
import os
from pyairtable import Api
from state.state import PCFParserState
from chains.workflow_chains import (
    store_rag_chain,
    save_log_chain
)
from tools.write_pcf_parser_record import update_pcf_parser_record_supabase_id
from graphiti_client import graphiti, initialize_indicies_and_constraints
from tools import store_to_graphiti as store_to_graphiti_mod
from tools.query_neo4j import Neo4jClient

logger = logging.getLogger(__name__)

def post_process_rag_node(state: PCFParserState) -> PCFParserState:
    """
    Node to store summaries to RAG.
    """
    created_parser_records = state.get("created_parser_records", [])
    
    logger.info("💾 STEP 5: Post-processing - Storing to RAG")
    
    rag_texts = [item.get("summary_text", "") for item in created_parser_records]
    rag_metadatas = [{"pcf_id": item.get("pcf_id")} for item in created_parser_records]

    rag_result = None
    rag_ids = []
    
    try:
        rag_result = store_rag_chain.invoke(
            {"texts": rag_texts, "metadatas": rag_metadatas}
        )
        
        if isinstance(rag_result, dict):
            rag_ids = rag_result.get("ids", [])
        
        save_log_chain.invoke(
            {
                "log_info": f"Stored {len(rag_texts)} summaries to RAG",
                "origin": "pcf_parser_workflow",
            }
        )
        
        # Update Airtable records with Supabase IDs
        if rag_ids and len(rag_ids) == len(created_parser_records):
            for idx, (item, supabase_id) in enumerate(zip(created_parser_records, rag_ids), 1):
                try:
                    record_id = item.get("created_record", {}).get("id")
                    if record_id and supabase_id:
                        update_pcf_parser_record_supabase_id(record_id, supabase_id)
                except Exception as update_err:
                    logger.error(f"Failed to update record with Supabase ID: {update_err}")
                    
    except Exception as e:
        logger.error(f"❌ Failed storing to RAG: {e}")
        save_log_chain.invoke(
            {
                "log_info": f"Failed storing to RAG: {e}",
                "origin": "pcf_parser_workflow",
            }
        )
        
    return {"rag_result": rag_result}

async def post_process_graphiti_node(state: PCFParserState) -> PCFParserState:
    """
    Node to store episodes to Graphiti.
    """
    created_parser_records = state.get("created_parser_records", [])
    logger.info(f"🕸️ Storing {len(created_parser_records)} episodes to Graphiti Knowledge Graph...")

    results = []
    try:
        try:
            await initialize_indicies_and_constraints()
        except Exception as ie:
            logger.warning(f"⚠️ Could not initialize Graphiti indices/constraints: {ie}")

        for idx, item in enumerate(created_parser_records, 1):
            pcf_id = item.get("pcf_id")
            record_id = item.get("created_record", {}).get("id")
            
            try:
                res = await store_to_graphiti_mod.add_episodes_to_graphiti(
                    core="Betafit",
                    domain="PCF Parser",
                    project=item.get("pcf_id", "unknown"),
                    component="Parser",
                    feature=(
                        item.get("created_record", {})
                        .get("fields", {})
                        .get("Name", "summary")
                    ),
                    description=(
                        item.get("created_record", {})
                        .get("fields", {})
                        .get("Summary", "")
                    ),
                )
                results.append(res)
                
                if res and record_id:
                    # Update Neo4j UUID
                    try:
                        neo4j_client = Neo4jClient(
                            uri=os.getenv("NEO4J_URI"),
                            user=os.getenv("NEO4J_USER"),
                            password=os.getenv("NEO4J_PASSWORD")
                        )
                        data = neo4j_client.get_latest_entity()
                        if data:
                            kg_uuid = data.get("b")["uuid"]
                            if kg_uuid:
                                api = Api(os.getenv("AIR_TABLE_ACCESS_TOKEN"))
                                table = api.table(
                                    os.getenv("AIR_TABLE_BASE_ID"),
                                    os.getenv("AIR_TABLE_PCF_PARSER_TABLE_ID")
                                )
                                table.update(record_id, {"KG UUID": kg_uuid})
                        neo4j_client.close()
                    except Exception as upd_err:
                        logger.error(f"Failed to query/update KG UUID: {upd_err}")
                        
            except Exception as e_inner:
                logger.error(f"Failed adding to Graphiti for {pcf_id}: {e_inner}")
                save_log_chain.invoke(
                    {
                        "log_info": f"Failed adding to Graphiti for {pcf_id}: {e_inner}",
                        "origin": "pcf_parser_workflow",
                    }
                )
                results.append(False)
                
            if idx < len(created_parser_records):
                await asyncio.sleep(10)
                
    finally:
        try:
            await graphiti.close()
        except Exception as close_e:
            logger.debug(f"⚠️ Could not close Graphiti client: {close_e}")
            
    return {"graphiti_result": results}
