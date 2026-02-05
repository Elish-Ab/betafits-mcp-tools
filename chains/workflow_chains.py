# moved from pcf parser repo
from langchain_core.runnables import RunnableLambda, RunnableConfig #type: ignore
from chains.tool_definitions import (
    fetch_pcf_table_tool,
    create_pcf_table_embeddings_tool,
    map_relevant_pcfs_tool,
    summarize_pcf_meeting_tool,
    create_pcf_parser_record_tool,
    store_to_rag_tool,
    save_log_tool
)

def fetch_pcf_records_wrapper(input_dict: dict, config: RunnableConfig):
    return fetch_pcf_table_tool.invoke(input_dict, config=config)

fetch_pcf_records_chain = RunnableLambda(fetch_pcf_records_wrapper)

def create_embeddings_wrapper(input_dict: dict, config: RunnableConfig):
    return create_pcf_table_embeddings_tool.invoke(input_dict, config=config)

create_embeddings_chain = RunnableLambda(create_embeddings_wrapper)

def map_pcfs_wrapper(input_dict: dict, config: RunnableConfig):
    return map_relevant_pcfs_tool.invoke(input_dict, config=config)

map_pcfs_chain = RunnableLambda(map_pcfs_wrapper)

def summarize_meeting_wrapper(input_dict: dict, config: RunnableConfig):
    return summarize_pcf_meeting_tool.invoke(input_dict, config=config)

summarize_meeting_chain = RunnableLambda(summarize_meeting_wrapper)

def create_record_wrapper(input_dict: dict, config: RunnableConfig):
    return create_pcf_parser_record_tool.invoke(input_dict, config=config)

create_record_chain = RunnableLambda(create_record_wrapper)

def store_rag_wrapper(input_dict: dict, config: RunnableConfig):
    return store_to_rag_tool.invoke(input_dict, config=config)

store_rag_chain = RunnableLambda(store_rag_wrapper)

def save_log_wrapper(input_dict: dict, config: RunnableConfig):
    return save_log_tool.invoke(input_dict, config=config)

save_log_chain = RunnableLambda(save_log_wrapper)
