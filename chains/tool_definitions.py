# moved from pcf parser repo
import json
import os
import logging
from langchain_core.tools import StructuredTool
from tools.fetch_pcf_table import fetch_pcf_table as fetch_pcf_table_func
from tools.save_logs import SaveLogInput, save_log_ as save_log_func
from tools.semantic_match import (
    CreateEmbeddingsInput,
    MapRelevantPCFsInput,
    create_pcf_table_embeddings,
    map_relevant_pcfs,
)
from tools.store_to_rag import (
    QueryRAGInput,
    StoreToRAGInput,
    query_rag_ as query_rag_func,
    store_to_rag_ as store_to_rag_func,
)
from tools.summarize import (
    SummarizeInput,
    SummarizePCFMeetingInput,
    summarize_pcf_meeting,
    summarize_text_ as summarize_text_func,
)
from tools.write_pcf_parser_record import (
    PCFParserRecordInput,
    create_pcf_parser_record,
)

# Configure logger
logger = logging.getLogger(__name__)

# Load tool configuration from config.json
# Assuming this file is in pcf_parser/chains/tool_definitions.py
# and config is in pcf_parser/data/config.json
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "config.json"))

TOOL_CONFIG = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        TOOL_CONFIG = json.load(f)["tools"]
else:
    logger.warning(f"Config file not found at {CONFIG_PATH}")

def create_tool_from_config(
    func, config_key: str, args_schema: type | None = None
) -> StructuredTool:
    """Create a StructuredTool from function using config.json metadata."""
    config = TOOL_CONFIG.get(config_key, {})
    return StructuredTool.from_function(
        func=func,
        name=config.get("name", config_key),
        description=config.get("description", ""),
        args_schema=args_schema,
        return_direct=True,
    )

# Create all StructuredTools using config
fetch_pcf_table_tool = create_tool_from_config(fetch_pcf_table_func, "fetch_pcf_table")

map_relevant_pcfs_tool = create_tool_from_config(
    map_relevant_pcfs, "map_relevant_pcfs", MapRelevantPCFsInput
)

create_pcf_table_embeddings_tool = create_tool_from_config(
    create_pcf_table_embeddings, "create_pcf_table_embeddings", CreateEmbeddingsInput
)

summarize_pcf_meeting_tool = create_tool_from_config(
    summarize_pcf_meeting, "summarize_pcf_meeting", SummarizePCFMeetingInput
)

summarize_text_tool = create_tool_from_config(
    summarize_text_func, "summarize_text", SummarizeInput
)

create_pcf_parser_record_tool = create_tool_from_config(
    create_pcf_parser_record, "create_pcf_parser_record", PCFParserRecordInput
)

store_to_rag_tool = create_tool_from_config(
    store_to_rag_func, "store_to_rag", StoreToRAGInput
)

query_rag_tool = create_tool_from_config(query_rag_func, "query_rag", QueryRAGInput)

save_log_tool = create_tool_from_config(save_log_func, "save_log", SaveLogInput)
