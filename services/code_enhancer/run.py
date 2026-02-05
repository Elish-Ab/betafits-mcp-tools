"""Service that enhances an existing repository with new functionality."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from lib.airtable_client import (
    get_all_repositories,
    get_all_folders,
    get_all_files,
)
from lib.code_generator_airtable_persistence import persist_repository_structure
from lib.llm_client import call_llm
from lib.validation import validate_user_message
from services.code_enhancer.prompt import build_enhancement_prompt
from services.code_reviewer.chain import build_repository_with_files
from services.code_generator.chain import extract_json_from_text


class RepositoryNotFoundError(ValueError):
    """Raised when an enhancement is requested for an unknown repository."""


def _find_repository_by_name(repo_name: str) -> Optional[Dict[str, Any]]:
    """Return the Airtable repo record whose Name matches repo_name (case-insensitive)."""
    for record in get_all_repositories():
        fields = record.get("fields", {})
        if fields.get("Name", "").strip().lower() == repo_name.strip().lower():
            return record
    return None


def _build_repo_snapshot(repo_record: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble folder/file context for the requested repository."""
    folders = get_all_folders()
    files = get_all_files()
    return build_repository_with_files(repo_record, folders, files)


def _plan_to_repository_structure(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Convert enhancement JSON into the repository structure shape expected by persistence."""
    repo_structure: Dict[str, Any] = {"repositories": [], "folders": [], "files": []}
    for folder in plan.get("folders", []) or []:
        path = folder.get("folder_path")
        if path:
            repo_structure["folders"].append(
                {
                    "folders.path": path.rstrip("/"),
                    "folders.description": folder.get("description", ""),
                }
            )
    for file_plan in plan.get("files", []) or []:
        folder_path = (file_plan.get("folder_path") or "").rstrip("/")
        file_name = file_plan.get("file_name")
        if not file_name:
            continue
        relative_path = f"{folder_path}/{file_name}" if folder_path else file_name
        repo_structure["files"].append(
            {
                "files.name": file_name,
                "files.path": relative_path,
                "files.language": file_plan.get("language"),
                "files.code": file_plan.get("source_code"),
                "files.rationale": file_plan.get("description"),
            }
        )
    return repo_structure


def enhance_repository(message: str, repo_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance an existing repository with new functionality.

    Args:
        message: User-provided enhancement request.
        repo_name: Airtable Name of the target repository.
        context: Reused generator context (field mappings, etc.).

    Returns:
        Dict describing the enhancement outcome and Airtable persistence summary.
    """
    validated_request = validate_user_message(message)
    repo_record = _find_repository_by_name(repo_name)
    if not repo_record:
        raise RepositoryNotFoundError(f"Repository '{repo_name}' not found in Airtable.")

    snapshot = _build_repo_snapshot(repo_record)
    prompt = build_enhancement_prompt(
        repo_name,
        validated_request,
        snapshot,
        context.get("langgraph_context", ""),
        context.get("pcf_context"),
        context.get("pcf_documents_summary", ""),
        context.get("extra_context_files_summary", ""),
    )
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    plan = extract_json_from_text(response)
    if not isinstance(plan, dict):
        raise ValueError("Enhancement LLM response did not contain valid JSON.")

    repo_structure = _plan_to_repository_structure(plan)
    if not repo_structure["files"]:
        raise ValueError("Enhancement plan did not produce any files to persist.")

    persistence_summary = persist_repository_structure(
        repo_structure,
        context.get("field_mappings", []),
        include_children=True,
        existing_repo_id=repo_record.get("id"),
    )

    return {
        "repository_id": repo_record.get("id"),
        "repository_name": repo_name,
        "enhancement_plan": plan,
        "airtable_persistence": persistence_summary,
    }
