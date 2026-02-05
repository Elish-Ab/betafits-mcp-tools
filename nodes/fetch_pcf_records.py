import logging
from state.state import PCFParserState
from chains.workflow_chains import (
    fetch_pcf_records_chain,
    create_embeddings_chain,
    save_log_chain
)

logger = logging.getLogger(__name__)

def fetch_pcf_records_node(state: PCFParserState) -> PCFParserState:
    """
    Node to fetch PCF records and create embeddings.
    """
    logger.info("📥 STEP 1: Fetching PCF records from Airtable")
    
    try:
        # Log start
        save_log_chain.invoke(
            {
                "log_info": "Starting PCF Parser workflow",
                "origin": "pcf_parser_workflow",
            }
        )

        records = fetch_pcf_records_chain.invoke({})
        logger.info(f"✓ Successfully fetched {len(records)} PCF records")
        
        save_log_chain.invoke(
            {
                "log_info": f"Fetched {len(records)} PCF records",
                "origin": "pcf_parser_workflow",
            }
        )
        
        # Create embeddings
        logger.info("🔢 STEP 2: Creating embeddings for semantic search")
        create_embeddings_chain.invoke({"records": records})
        logger.info("✓ Successfully created embeddings and BM25 index")
        
        save_log_chain.invoke(
            {
                "log_info": "Created PCF embeddings and BM25 index",
                "origin": "pcf_parser_workflow",
            }
        )
        
        return {"pcf_records": records}
        
    except Exception as e:
        logger.error(f"❌ Failed in fetch_pcf_records_node: {e}")
        save_log_chain.invoke(
            {
                "log_info": f"Failed in fetch_pcf_records_node: {e}",
                "origin": "pcf_parser_workflow",
            }
        )
        return {"errors": [str(e)]}
