"""Prompt builder for repository enhancement requests."""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional


def _summarize_folders(folders: List[Dict[str, Any]]) -> str:
    lines = []
    for folder in folders[:15]:
        fields = folder.get("fields", {})
        path = fields.get("Folder Path") or fields.get("folders.path") or "(unknown path)"
        description = fields.get("Description") or "No description"
        lines.append(f"- {path}: {description}")
    if not lines:
        return "No folders found."
    return "\n".join(lines)


def _summarize_files(files: List[Dict[str, Any]]) -> str:
    lines = []
    for file_record in files[:15]:
        fields = file_record.get("fields", {})
        file_path = fields.get("Relative File Path") or fields.get("files.path") or "(unknown)"
        language = fields.get("Language") or "Unknown language"
        lines.append(f"- {file_path} [{language}]")
    if not lines:
        return "No files found."
    return "\n".join(lines)


def build_enhancement_prompt(
    repo_name: str,
    enhancement_request: str,
    repo_snapshot: Dict[str, Any],
    langgraph_context: str = "",
    pcf_context: Optional[Dict[str, Any]] = None,
    pcf_documents_summary: str = "",
    extra_context_files_summary: str = "",
) -> str:
    """Create the prompt describing how to extend an existing repository."""
    folder_summary = _summarize_folders(repo_snapshot.get("folders", []))
    file_summary = _summarize_files(repo_snapshot.get("files", []))

    return f"""You are enhancing the existing Betafits repository named "{repo_name}".

All changes must follow the engineering standard (apps/, services/, workflows/, lib/, tests/). Only describe NEW folders/files that are required for the requested feature. When updating existing functionality, prefer creating helper modules/services rather than rewriting core files.

Existing folder structure:
{folder_summary}

Representative files:
{file_summary}

LangGraph context (optional):
{langgraph_context or "None provided."}

PCF context (optional):
{json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."}

PCF documents (optional):
{pcf_documents_summary or "None provided."}

Additional file context (optional):
{extra_context_files_summary or "None provided."}

Enhancement request:
{enhancement_request}

Return STRICT JSON with the following schema (and nothing else):
{{
  "summary": "High-level summary of the enhancement",
  "folders": [
    {{
      "folder_path": "relative/path",
      "description": "purpose of the folder"
    }}
  ],
  "files": [
    {{
      "folder_path": "services/new_feature",
      "file_name": "handler.py",
      "language": "Python",
      "description": "what the file does",
      "source_code": "complete file contents"
    }}
  ]
}}

Guidelines:
- Only include folders/files that need to be CREATED as part of this enhancement.
- Respect snake_case naming and Betafits coding conventions.
- Source code must be fully self-contained and runnable without additional context.
- If no new folders are required, return an empty list for "folders".
"""
