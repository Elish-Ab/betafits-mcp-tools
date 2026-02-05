import logging
from state.state import PCFParserState
from chains.workflow_chains import (
    map_pcfs_chain,
    save_log_chain
)

logger = logging.getLogger(__name__)

def map_pcfs_node(state: PCFParserState) -> PCFParserState:
    """
    Node to map relevant PCFs based on the transcript.
    """
    transcript = state.get("transcript", "")
    top_k = state.get("top_k", 5)
    
    logger.info("🔍 STEP 3: Mapping relevant PCFs using semantic search + LLM re-ranking")
    
    try:
        mapping = map_pcfs_chain.invoke(
            {"transcript": transcript, "top_k": top_k, "filter_current": True}
        )
        mapped = mapping.get("relevant_pcfs", [])
        
        logger.info(f"✓ Found {len(mapped)} relevant PCF(s)")
        
        if not mapped:
            logger.warning("⚠️ No matching PCFs found")
            save_log_chain.invoke(
                {
                    "log_info": "No matching PCFs found; stopping (MVP behaviour).",
                    "origin": "pcf_parser_workflow",
                }
            )
            
        return {"mapped_pcfs": mapped}
        
    except Exception as e:
        logger.error(f"❌ Failed in map_pcfs_node: {e}")
        save_log_chain.invoke(
            {
                "log_info": f"Failed in map_pcfs_node: {e}",
                "origin": "pcf_parser_workflow",
            }
        )
        return {"errors": [str(e)]}
