"""Tool registry for Airtable context fetching tools."""
from typing import Any, Dict, List
from lib.airtable_client import (
    get_repositories_fields,
    get_folders_fields,
    get_files_fields,
    get_code_generator_output_format,
    get_all_field_mappings,
    get_field_mappings_for_code_generator,
    get_field_mappings_for_code_reviewer,
    get_all_libraries,
    get_all_repositories,
    get_all_folders,
    get_all_files,
    get_code_reviewer_output_format,
)


# Tool descriptions for LLM to understand available tools
TOOL_DESCRIPTIONS = {
    "get_repositories_fields": {
        "description": "Get fields and types from the repositories table in Airtable.",
        "parameters": {}
    },
    "get_folders_fields": {
        "description": "Get fields and types from the folders table in Airtable.",
        "parameters": {}
    },
    "get_files_fields": {
        "description": "Get fields and types from the files table in Airtable.",
        "parameters": {}
    },
    "get_code_generator_output_format": {
        "description": "Fetch data from the Io Formats table where name is 'Code Generator Output (Github Extractor)'. Returns actual record data with all fields.",
        "parameters": {}
    },
    "get_all_field_mappings": {
        "description": "Fetch all records from the Field Mappings table in Airtable. Returns all records with their fields.",
        "parameters": {}
    },
    "get_field_mappings_for_code_generator": {
        "description": "Fetch Field Mappings filtered by the Code Generator Output IO Format. Automatically filters to only return Field Mappings linked to the Code Generator Output (Github Extractor) IO Format.",
        "parameters": {}
    },
    "get_field_mappings_for_code_reviewer": {
        "description": "Fetch Field Mappings filtered by the Code Reviewer Output IO Format. Used to shape the code reviewer JSON output to match Airtable.",
        "parameters": {}
    },
    "get_all_libraries": {
        "description": "Fetch all records from the Libraries table in Airtable. Returns all libraries with their preferred/non-preferred status.",
        "parameters": {}
    },
    "get_all_repositories": {
        "description": "Fetch all records from the Repositories table in Airtable. Returns each repository with its record ID and all fields.",
        "parameters": {}
    },
    "get_all_folders": {
        "description": "Fetch all records from the Folders table in Airtable. Returns each folder with its record ID and all fields.",
        "parameters": {}
    },
    "get_all_files": {
        "description": "Fetch all records from the Files table in Airtable. Returns each file with its record ID and all fields.",
        "parameters": {}
    },
    "get_code_reviewer_output_format": {
        "description": "Fetch data from the Io Formats table where name is 'Code Reviewer Output'. Defines the expected structure for code reviewer output.",
        "parameters": {}
    },
}


def call_tool(tool_name: str, **kwargs) -> Any:
    """
    Call a registered tool by name.
    
    Args:
        tool_name: Name of the tool to call
        **kwargs: Arguments to pass to the tool (not used for field functions)
        
    Returns:
        Dictionary mapping field names to their types
    """
    if tool_name == "get_repositories_fields":
        return get_repositories_fields()
    elif tool_name == "get_folders_fields":
        return get_folders_fields()
    elif tool_name == "get_files_fields":
        return get_files_fields()
    elif tool_name == "get_code_generator_output_format":
        return get_code_generator_output_format()
    elif tool_name == "get_all_field_mappings":
        return get_all_field_mappings()
    elif tool_name == "get_field_mappings_for_code_generator":
        return get_field_mappings_for_code_generator()
    elif tool_name == "get_field_mappings_for_code_reviewer":
        return get_field_mappings_for_code_reviewer()
    elif tool_name == "get_all_libraries":
        return get_all_libraries()
    elif tool_name == "get_all_repositories":
        return get_all_repositories()
    elif tool_name == "get_all_folders":
        return get_all_folders()
    elif tool_name == "get_all_files":
        return get_all_files()
    elif tool_name == "get_code_reviewer_output_format":
        return get_code_reviewer_output_format()
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def get_tool_descriptions() -> Dict[str, Dict[str, Any]]:
    """
    Get descriptions of all available tools.
    
    Returns:
        Dictionary mapping tool names to their descriptions
    """
    return TOOL_DESCRIPTIONS
