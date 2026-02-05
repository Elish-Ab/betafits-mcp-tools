"""Main entry point for code generator service."""
from typing import Dict, Any, Optional
from lib.context_engine import build_generator_context, DEFAULT_PCF_TABLE
from services.code_generator.prompt import build_code_generation_prompt
from services.code_generator.io import parse_output, format_output_for_airtable, save_output_to_files
from lib.code_generator_airtable_persistence import persist_repository_structure
from services.code_generator.chain import run_code_generator_chain
from lib.validation import validate_user_message
from lib.llm_client import call_llm
from services.code_enhancer.run import enhance_repository, RepositoryNotFoundError


def prepare_context(
    message: str,
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
    context_files: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Prepare all context needed for code generation.

    Args:
        message: User's input message.
        pcf_record_id: Optional Airtable record ID for PCF context.
        pcf_table: Airtable table name containing the PCF record.
        context_files: Optional list of file paths to inject as context.

    Returns:
        Dictionary with all context information.
    """
    return build_generator_context(
        message,
        pcf_record_id=pcf_record_id,
        pcf_table=pcf_table,
        context_files=context_files,
    )


def run(
    message: str,
    use_chain: bool = True,
    persist: bool = True,
    repo_name: Optional[str] = None,
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
    context_files: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Run the code generator service.
    
    Args:
        message: User's input message
        use_chain: Whether to use LangChain pipeline (default: True)
        persist: Whether to persist output to Airtable when supported.
        repo_name: Optional existing repository name for enhancement mode.
        pcf_record_id: Optional Airtable record ID for PCF context.
        pcf_table: Airtable table name containing the PCF record.
        
    Returns:
        Dictionary with generated output
    """
    validated_message = validate_user_message(message)

    # Prepare context
    context = prepare_context(
        validated_message,
        pcf_record_id=pcf_record_id,
        pcf_table=pcf_table,
        context_files=context_files,
    )

    if repo_name:
        return enhance_repository(validated_message, repo_name, context)
    
    if use_chain:
        # Use multi-step chain
        chain_result = run_code_generator_chain(context)
        repository_structure = chain_result.get("repository", {})
        persistence_summary = {}
        if persist and repository_structure:
            try:
                persistence_summary = persist_repository_structure(
                    repository_structure,
                    context.get("field_mappings", []),
                    include_children=True,
                )
            except Exception as exc:
                print(f"Warning: Failed to persist generated repository to Airtable: {exc}")
        
        # Return the complete repository structure in JSON (no file creation)
        result = {
            "srs": chain_result.get("srs", {}),
            "architecture": chain_result.get("architecture", {}),
            "repository": chain_result.get("repository", {}),  # Complete repository with all code in JSON
            "summary": chain_result.get("summary", {}),
             "airtable_persistence": persistence_summary,
            "context_used": {
                "repositories_fields_count": len(context.get("repositories_fields", {})),
                "folders_fields_count": len(context.get("folders_fields", {})),
                "files_fields_count": len(context.get("files_fields", {})),
                "io_format_present": bool(context.get("io_format")),
                "field_mappings_count": len(context.get("field_mappings", [])),
                "preferred_libraries_count": len(context.get("preferred_libraries", [])),
                "non_preferred_libraries_count": len(context.get("non_preferred_libraries", [])),
                "coding_style_length": len(context.get("coding_style", "")),
                "engineering_standard_length": len(context.get("engineering_standard", "")),
                "langgraph_context_length": len(context.get("langgraph_context", "")),
                "pcf_context_present": bool(context.get("pcf_context")),
                "pcf_context_error": context.get("pcf_context_error"),
                "pcf_documents_count": len(context.get("pcf_documents", [])),
                "documents_context_error": context.get("documents_context_error"),
            },
        }
    else:
        # Fallback to direct LLM call
        prompt = build_code_generation_prompt(context)
        raw_output = call_llm(prompt, model="google/gemini-2.0-flash-001")
        
        # Parse output
        parsed_output = parse_output(raw_output)
        
        # Save files to disk
        saved_files = []
        try:
            output_dir = Path("generated_repository")
            saved_files = save_output_to_files(parsed_output, output_dir)
        except Exception as e:
            print(f"Warning: Could not save files to disk: {e}")
        
        result = {
            "output": parsed_output,
            "formatted_for_airtable": format_output_for_airtable(parsed_output),
            "saved_files": [str(f) for f in saved_files],
            "context_used": {
                "repositories_fields_count": len(context.get("repositories_fields", {})),
                "folders_fields_count": len(context.get("folders_fields", {})),
                "files_fields_count": len(context.get("files_fields", {})),
                "io_format_present": bool(context.get("io_format")),
                "coding_style_length": len(context.get("coding_style", "")),
                "engineering_standard_length": len(context.get("engineering_standard", "")),
                "langgraph_context_length": len(context.get("langgraph_context", "")),
                "pcf_context_present": bool(context.get("pcf_context")),
                "pcf_context_error": context.get("pcf_context_error"),
                "pcf_documents_count": len(context.get("pcf_documents", [])),
                "documents_context_error": context.get("documents_context_error"),
            },
        }
    
    return result
