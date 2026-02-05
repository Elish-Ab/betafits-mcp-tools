"""Helpers for writing code reviewer results back to Airtable."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from lib.airtable_client import (
    batch_update_files,
    batch_update_folders,
    batch_update_repositories,
)

SKIP_FIELD_NAMES = {"repo_id", "folder_id", "file_id", "id"}
IMPROVEMENT_KEYWORDS = {
    "Improve Error Handling": ["error handling", "error", "exception"],
    "Add Input Validation": ["input validation", "validate", "validation"],
    "Refactor for Readability": ["refactor", "readability", "maintainability"],
    "Add Documentation": ["documentation", "docstring", "comment"],
    "Add Logging": ["logging", "log"],
    "Modularize Code": ["modularize", "modular", "reusable", "module"],
    "Improve Performance": ["performance", "optimize", "optimization"],
    "Add Tests": ["test", "unit test", "coverage"],
    "Enhance Security": ["security", "auth", "authentication", "authorization"],
    "Secure Secrets": ["secret", "api key", "credential", "hardcode", "hard-coded"],
    "Manage Dummy Data": ["dummy data", "sample data", "fixture"],
    "Use Native APIs": ["node.js syntax", "require", "process.env", "urlfetchapp", "propertieservice"],
}
ENTITY_KEYWORDS = {
    "repositories": ("repository", "repositories"),
    "folders": ("folder", "folders"),
    "files": ("file", "files"),
}


def persist_review_results(review: Dict[str, Any], field_mappings: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Persist review results to Airtable using the provided field mappings.

    Args:
        review: Review payload returned by the code reviewer chain.
        field_mappings: Field Mapping records (Action Type = Populate).

    Returns:
        Dictionary summarizing how many records were updated per table.
    """
    if not review or not field_mappings:
        return {}

    grouped_mappings = _group_field_mappings(field_mappings)
    repo_reviews = [review] if isinstance(review, dict) else []

    repo_updates = _build_updates(
        repo_reviews,
        grouped_mappings.get("repositories", []),
        id_field="repo_id",
    )
    folder_updates = _build_updates(
        review.get("folders", []) if isinstance(review, dict) else [],
        grouped_mappings.get("folders", []),
        id_field="folder_id",
    )
    file_updates = _build_updates(
        review.get("files", []) if isinstance(review, dict) else [],
        grouped_mappings.get("files", []),
        id_field="file_id",
    )

    summary: Dict[str, int] = {}
    if repo_updates:
        batch_update_repositories(repo_updates)
        summary["repositories"] = len(repo_updates)
    if folder_updates:
        batch_update_folders(folder_updates)
        summary["folders"] = len(folder_updates)
    if file_updates:
        batch_update_files(file_updates)
        summary["files"] = len(file_updates)

    return summary


def _group_field_mappings(field_mappings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {"repositories": [], "folders": [], "files": []}
    for mapping in field_mappings:
        fields = mapping.get("fields", {})
        entity = _detect_entity(fields)
        if entity:
            grouped[entity].append(fields)
    return grouped


def _detect_entity(fields: Dict[str, Any]) -> str:
    json_pattern = (fields.get("JSON Pattern") or "").lower()
    current_name = (fields.get("Current Name Field") or "").lower()

    for entity, keywords in ENTITY_KEYWORDS.items():
        if any(keyword in json_pattern for keyword in keywords):
            return entity
        if any(keyword in current_name for keyword in keywords):
            return entity
    return "repositories"


def _build_updates(
    records: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    id_field: str,
) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    if not records or not mappings:
        return updates

    for record in records:
        record_id = record.get(id_field)
        if not record_id:
            continue
        fields = _build_fields(record, mappings)
        if fields:
            updates.append({"id": record_id, "fields": fields})
    return updates


def _build_fields(record: Dict[str, Any], mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for mapping in mappings:
        field_name = mapping.get("Current Name Field")
        if not field_name or field_name in SKIP_FIELD_NAMES:
            continue
        value = _extract_value(record, mapping)
        if value is None:
            continue
        if field_name == "Room for Improvement":
            normalized = _normalize_room_for_improvement(value)
            if not normalized:
                continue
            fields[field_name] = normalized
            continue
        fields[field_name] = _normalize_value(value)
    return fields


def _extract_value(record: Dict[str, Any], mapping: Dict[str, Any]) -> Any:
    field_name = mapping.get("Current Name Field", "")
    value = _get_nested_value(record, field_name)
    if value is not None:
        return value

    json_pattern = mapping.get("JSON Pattern", "")
    relative_path = _relative_path(json_pattern)
    if relative_path:
        return _get_nested_value(record, relative_path)
    return None


def _relative_path(pattern: str) -> str:
    if not pattern:
        return ""
    cleaned = pattern.strip()
    cleaned = cleaned.replace("[]", "")
    cleaned = re.sub(r"^review\.", "", cleaned, flags=re.IGNORECASE)
    if "]." in cleaned:
        cleaned = cleaned.split("].")[-1]
    return cleaned


def _get_nested_value(data: Any, path: str) -> Any:
    if not path:
        return None
    if isinstance(data, dict) and path in data:
        return data[path]

    tokens = [token for token in path.split(".") if token]
    current: Any = data
    for token in tokens:
        if isinstance(current, dict):
            current = _dict_value(current, token)
        else:
            return None
    return current


def _dict_value(data: Dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]

    lowered = key.lower()
    for candidate_key, candidate_value in data.items():
        if candidate_key.lower() == lowered:
            return candidate_value

    normalized_target = _normalize_key(key)
    for candidate_key, candidate_value in data.items():
        if _normalize_key(candidate_key) == normalized_target:
            return candidate_value

    return None


def _normalize_key(key: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_")


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def persist_repository_core_fields(review: Dict[str, Any]) -> bool:
    """
    Persist essential repository evaluation fields (grades, summary, recommendations)
    directly to the Repositories table.
    """
    if not review or not review.get("repo_id"):
        return False

    grades = review.get("grades", {}) or {}
    suggestions = review.get("suggestions") or []

    fields = {
        "Review Summary": review.get("summary"),
        "Repo Structure Grade": grades.get("repo_structure"),
        "Code Quality Grade": grades.get("code_quality"),
        "Documentation Grade": grades.get("documentation"),
        "Security Practices Grade": grades.get("security_practices"),
        "Modularity/Testability Grade": grades.get("modularity_testability"),
        "Dependency Mgmt": grades.get("dependency_management"),
        "Room for Improvement": _normalize_room_for_improvement(review.get("room_for_improvement")),
        "Performance": grades.get("performance"),
        "Error Handling": grades.get("error_handling"),
        "Style Guide Compliance": grades.get("style_guide_compliance"),
        "Recommendations": "\n".join(suggestions) if suggestions else None,
    }
    cleaned = {k: v for k, v in fields.items() if v not in (None, "")}
    if not cleaned:
        return False

    payload = {"id": review["repo_id"], "fields": cleaned}
    return _safe_repository_update(payload)


def _safe_repository_update(record: Dict[str, Any]) -> bool:
    fields = record.get("fields", {}).copy()
    if not fields:
        return False

    while fields:
        try:
            batch_update_repositories([{"id": record["id"], "fields": fields}])
            return True
        except Exception as exc:
            unknown_field = _extract_unknown_field_name(exc)
            if unknown_field and unknown_field in fields:
                print(f"  Warning: Repository column '{unknown_field}' not found in Airtable. Skipping it.")
                fields.pop(unknown_field, None)
                continue
            raise
    return False


def _extract_unknown_field_name(exc: Exception) -> Optional[str]:
    message = str(exc)
    match = re.search(r'Unknown field name:\s*"([^"]+)"', message)
    if match:
        return match.group(1)
    return None


def persist_files_core_fields(review: Dict[str, Any]) -> int:
    files = review.get("files") or []
    if not isinstance(files, list):
        return 0

    updates = []
    for file_data in files:
        file_id = file_data.get("file_id")
        if not file_id:
            continue
        fields = {}

        language = file_data.get("language")
        if language:
            fields["Language"] = language

        loc = file_data.get("lines_of_code")
        if loc is not None:
            fields["Lines of Code"] = loc

        bugginess = file_data.get("bugginess")
        if isinstance(bugginess, str) and bugginess:
            fields["Bugginess"] = bugginess.capitalize()

        improvement = file_data.get("room_for_improvement")
        improvement_text = _normalize_room_for_improvement(improvement)
        if improvement_text:
            fields["Room for Improvement"] = improvement_text

        if fields:
            updates.append({"id": file_id, "fields": fields})

    updated_count = 0
    for record in updates:
        if _safe_file_update(record):
            updated_count += 1
    return updated_count


def _safe_file_update(record: Dict[str, Any]) -> bool:
    fields = record.get("fields", {}).copy()
    if not fields:
        return False

    while fields:
        try:
            batch_update_files([{"id": record["id"], "fields": fields}])
            return True
        except Exception as exc:
            message = str(exc)
            if "Insufficient permissions to create new select option" in message:
                if "Room for Improvement" in fields:
                    print("  Warning: Room for Improvement options not allowed. Skipping field for this file.")
                    fields.pop("Room for Improvement", None)
                    continue
            unknown_field = _extract_unknown_field_name(exc)
            if unknown_field and unknown_field in fields:
                print(f"  Warning: File column '{unknown_field}' not found in Airtable. Skipping it.")
                fields.pop(unknown_field, None)
                continue
            raise
    return False


def _normalize_room_for_improvement(value: Any) -> Optional[str]:
    texts: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                texts.append(item)
    elif isinstance(value, str) and value.strip():
        texts.append(value)

    options = _map_improvements_to_options(texts)
    if not options:
        return None
    if len(options) == 1:
        return options[0]
    return "; ".join(options)


def _map_improvements_to_options(texts: List[str]) -> List[str]:
    if not texts:
        return []

    selected: List[str] = []
    for text in texts:
        lower = text.lower()
        if lower.strip() in {"n/a", "none", "na", "not applicable"}:
            continue
        matched = False
        for option, keywords in IMPROVEMENT_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                selected.append(option)
                matched = True

        if not matched:
            if any(token in lower for token in ["dummy data", "sample data", "fixture"]):
                selected.append("Manage Dummy Data")
            elif any(token in lower for token in ["node.js", "require(", "process.env", "urlfetchapp", "propertiesservice"]):
                selected.append("Use Native APIs")
            elif "hardcod" in lower or "hard-coded" in lower or "sensitive" in lower or "propertiesservice" in lower:
                selected.append("Secure Secrets")
    # Remove duplicates while preserving order
    seen = set()
    unique_options = []
    for option in selected:
        if option not in seen:
            seen.add(option)
            unique_options.append(option)
    return unique_options
