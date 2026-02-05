"""Helpers to persist generated repository structures into Airtable."""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Set

from lib.airtable_client import (
    batch_create_files,
    batch_create_folders,
    batch_create_repositories,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

REPO_LINK_FIELD = "Repositories (Link to Repositories)"
FOLDER_LINK_FIELD = "Folders (Link To Folders)"

# Fallback mappings if Field Mappings are missing or incomplete
DEFAULT_REPO_FIELDS = [
    ("repositories.name", "Name"),
    ("repositories.description", "Description"),
]

DEFAULT_FOLDER_FIELDS = [
    ("folders.path", "Folder Path"),
    ("folders.description", "Description"),
]

DEFAULT_FILE_FIELDS = [
    ("files.name", "File Name"),
    ("files.language", "Language"),
    ("files.code", "Source Code"),
    ("Source Code", "Source Code"),
    ("files.rationale", "Description"),
]

FIELD_ALIASES = {
    "Reasoning / Rationale": "Description",
}

DISALLOWED_TARGET_FIELDS = {
    "Libraries (Link to Libraries)",
    "Relative File Path",
}

ALLOWED_REPO_FIELDS = {dest for _, dest in DEFAULT_REPO_FIELDS}

ALLOWED_LANGUAGES = {
    "python": "Python",
    "py": "Python",
    "apps script": "Apps Script",
    "google apps script": "Apps Script",
    "appscript": "Apps Script",
    "javascript": "Java Script",
    "java script": "Java Script",
    "js": "Java Script",
}

EXTENSION_LANGUAGE_MAP = {
    ".py": "Python",
    ".gs": "Apps Script",
    ".js": "Java Script",
}


def persist_repository_structure(
    repository: Dict[str, Any],
    field_mappings: List[Dict[str, Any]],
    include_children: bool = False,
    existing_repo_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Persist the generated repository/folder/file structure to Airtable.

    Returns:
        Dictionary summarizing how many records were created per table.
    """
    if not repository:
        return {}

    grouped = _group_field_mappings(field_mappings or [])

    repo_id = existing_repo_id
    summary = {}

    if not existing_repo_id:
        repos_data = repository.get("repositories", []) or []
        repo_payloads, repo_meta = _build_repo_payloads(repos_data, grouped["repositories"])
        _log_payload_fields("repositories", repo_payloads)
        created_repos = batch_create_repositories(repo_payloads) if repo_payloads else []
        repo_ids = [rec.get("id") for rec in created_repos]
        repo_id = repo_ids[0] if repo_ids else None

        if repo_payloads:
            summary["repositories"] = len(created_repos)

    if include_children and repo_id:
        folders_data = repository.get("folders", []) or []
        folder_payloads, folder_meta = _build_folder_payloads(folders_data, grouped["folders"], repo_id)
        _log_payload_fields("folders", folder_payloads)
        created_folders = batch_create_folders(folder_payloads) if folder_payloads else []
        folder_id_by_path = {}
        for meta, record in zip(folder_meta, created_folders):
            path_key = meta.get("path")
            if path_key:
                folder_id_by_path[path_key] = record.get("id")
        if folder_payloads:
            summary["folders"] = len(created_folders)

        files_data = repository.get("files", []) or []
        file_payloads = _build_file_payloads(files_data, grouped["files"], folder_id_by_path)
        _log_payload_fields("files", file_payloads)
        created_files = batch_create_files(file_payloads) if file_payloads else []
        if file_payloads:
            summary["files"] = len(created_files)

    return summary


def _group_field_mappings(field_mappings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    grouped = {"repositories": [], "folders": [], "files": []}
    for mapping in field_mappings:
        fields = mapping.get("fields", {})
        json_key = fields.get("Current Name Field")
        target_field = _extract_target_field_name(fields)
        entity = _detect_entity(fields)
        if not json_key or not target_field:
            continue
        if entity == "repositories" and target_field not in ALLOWED_REPO_FIELDS:
            continue
        grouped[entity].append({
            "json_key": json_key,
            "target_field": target_field,
        })

    if not grouped["repositories"]:
        grouped["repositories"] = [{"json_key": src, "target_field": dest} for src, dest in DEFAULT_REPO_FIELDS]
    if not grouped["folders"]:
        grouped["folders"] = [{"json_key": src, "target_field": dest} for src, dest in DEFAULT_FOLDER_FIELDS]
    if not grouped["files"]:
        grouped["files"] = [{"json_key": src, "target_field": dest} for src, dest in DEFAULT_FILE_FIELDS]

    return grouped


def _build_repo_payloads(
    records: List[Dict[str, Any]],
    mappings: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payloads: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    for record in records:
        fields = _apply_mappings(record, mappings)
        name = _sanitize_repo_name(fields.get("Name"))
        if not name:
            continue
        fields["Name"] = name
        payloads.append({"fields": fields})
        meta.append({"raw": record})
    return payloads, meta


def _log_payload_fields(entity: str, payloads: List[Dict[str, Any]]) -> None:
    """Log the Airtable fields for each payload before sending the request."""
    if not payloads:
        return
    for idx, payload in enumerate(payloads):
        field_values = payload.get("fields") or {}
        field_names = sorted(field_values.keys())
        logger.info("Airtable payload[%s][%s] fields=%s payload=%s", entity, idx, field_names, field_values)


def _extract_target_field_name(fields: Dict[str, Any]) -> Optional[str]:
    candidates = fields.get("Field Name (from Target Field (Link to Data Dictionary))")
    if isinstance(candidates, list) and candidates:
        value = candidates[0]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped.lower() not in {"fields", "field"}:
                return stripped
        return None
    fallback = fields.get("Current Name Field")
    if isinstance(fallback, str):
        stripped = fallback.strip()
        if stripped and stripped.lower() not in {"fields", "field"}:
            return stripped
    return None


def _detect_entity(fields: Dict[str, Any]) -> str:
    json_pattern = (fields.get("JSON Pattern") or "").lower()
    current_name = (fields.get("Current Name Field") or "").lower()
    if "folder" in json_pattern or "folders" in json_pattern or "folder" in current_name:
        return "folders"
    if "file" in json_pattern or "files" in json_pattern or "file" in current_name:
        return "files"
    return "repositories"


def _build_entity_payloads(
    records: List[Dict[str, Any]],
    mappings: List[Dict[str, str]],
    allowed_fields: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payloads = []
    meta = []
    for record in records:
        fields = _apply_mappings(record, mappings, allowed_fields)
        if not fields:
            continue
        payloads.append({"fields": fields})
        meta.append({"raw": record})
    return payloads, meta


def _build_folder_payloads(
    folders: List[Dict[str, Any]],
    mappings: List[Dict[str, str]],
    repo_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payloads = []
    meta = []
    for folder in folders:
        fields = _apply_mappings(folder, mappings)
        if repo_id:
            fields.setdefault(REPO_LINK_FIELD, [repo_id])
        if "Description" not in fields or not fields.get("Description"):
            description = _folder_description_from_path(folder)
            if description:
                fields["Description"] = description
        if not fields:
            continue
        payloads.append({"fields": fields})
        meta.append({"path": _folder_path_key(folder)})
    return payloads, meta


def _build_file_payloads(files: List[Dict[str, Any]], mappings: List[Dict[str, str]], folder_ids: Dict[str, str]) -> List[Dict[str, Any]]:
    payloads = []
    for file_record in files:
        fields = _apply_mappings(file_record, mappings)
        if not fields:
            continue
        if "File Name" not in fields or not fields.get("File Name"):
            inferred_name = _infer_file_name(file_record)
            if inferred_name:
                fields["File Name"] = inferred_name
        normalized_language = _normalize_language(fields.get("Language"), file_record)
        if normalized_language:
            fields["Language"] = normalized_language
        else:
            fields.pop("Language", None)
        if "Description" not in fields or not fields.get("Description"):
            description = file_record.get("files.rationale") or file_record.get("Description")
            if description:
                fields["Description"] = description
        source_code = fields.get("Source Code")
        if source_code is None:
            source_code = file_record.get("files.code") or file_record.get("Source Code")
            if source_code is not None:
                fields["Source Code"] = source_code
        lines_of_code = _compute_lines_of_code(source_code)
        if lines_of_code is not None:
            fields["Lines of Code"] = lines_of_code
        folder_key = _infer_folder_key(file_record)
        folder_id = folder_ids.get(folder_key)
        if folder_id:
            fields.setdefault(FOLDER_LINK_FIELD, [folder_id])
        payloads.append({"fields": fields})
    return payloads


def _apply_mappings(
    record: Dict[str, Any],
    mappings: List[Dict[str, str]],
    allowed_fields: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for mapping in mappings:
        json_key = mapping.get("json_key")
        target_field = mapping.get("target_field")
        if not json_key or not target_field:
            continue
        target_field_clean = target_field.strip()
        if target_field_clean in DISALLOWED_TARGET_FIELDS:
            continue
        if allowed_fields and target_field_clean not in allowed_fields:
            continue
        value = _get_value(record, json_key)
        if value is None or value == "":
            continue
        if target_field_clean.lower() in {"fields", "field"}:
            continue
        alias = FIELD_ALIASES.get(target_field_clean)
        final_field = alias or target_field_clean
        fields[final_field] = value
    return fields


def _get_value(record: Dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    shorthand = key.split(".")[-1]
    if shorthand in record:
        return record[shorthand]
    snake_key = key.replace(".", "_")
    if snake_key in record:
        return record[snake_key]
    for candidate_key, candidate_value in record.items():
        normalized = candidate_key.replace(".", " ").replace("_", " ").lower()
        if normalized.strip() == key.replace(".", " ").replace("_", " ").lower().strip():
            return candidate_value
    return None


def _folder_path_key(folder: Dict[str, Any]) -> Optional[str]:
    for key in ("folders.path", "Folder Path", "Path", "path"):
        value = folder.get(key)
        if value:
            return _normalize_folder_path(value)
    return None


def _infer_folder_key(file_record: Dict[str, Any]) -> Optional[str]:
    for key in ("files.path", "Relative File Path", "path"):
        path = file_record.get(key)
        if path:
            parent = PurePosixPath(path).parent.as_posix()
            if parent == ".":
                parent = ""
            return _normalize_folder_path(parent + ("/" if parent and not parent.endswith("/") else ""))
    return None


def _normalize_folder_path(path: str) -> str:
    normalized = path.strip()
    if normalized and not normalized.endswith("/") and "." not in normalized.split("/")[-1]:
        normalized += "/"
    return normalized


def _folder_description_from_path(folder: Dict[str, Any]) -> Optional[str]:
    path = folder.get("folders.description") or folder.get("folders.path") or folder.get("Folder Path") or folder.get("path")
    if not path:
        return None
    return f"Autogenerated folder for {path}" 


def _infer_file_name(file_record: Dict[str, Any]) -> Optional[str]:
    for key in ("files.name", "File Name", "files.path", "Relative File Path", "path"):
        value = file_record.get(key)
        if isinstance(value, str) and value.strip():
            if "/" in value:
                return value.rsplit("/", 1)[-1]
            return value
    return None


def _compute_lines_of_code(source_code: Optional[str]) -> Optional[int]:
    if not isinstance(source_code, str):
        return None
    lines = [line for line in source_code.splitlines() if line.strip()]
    return len(lines)


def _normalize_language(language: Optional[str], file_record: Dict[str, Any]) -> Optional[str]:
    if isinstance(language, str):
        normalized = ALLOWED_LANGUAGES.get(language.strip().lower())
        if normalized:
            return normalized

    path = file_record.get("files.path") or file_record.get("Relative File Path") or file_record.get("path")
    if isinstance(path, str):
        extension = "" if "." not in path else "." + path.rsplit(".", 1)[-1]
        normalized = EXTENSION_LANGUAGE_MAP.get(extension.lower())
        if normalized:
            return normalized

    return None


def _sanitize_repo_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    normalized = re.sub(r"[^a-z0-9-]", "-", name.strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or None
