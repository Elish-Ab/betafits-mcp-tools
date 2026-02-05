"""Central context engine for shared Airtable and docs context loading."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyairtable import Api

from lib.airtable_client import get_record_by_id
from lib.config import AIRTABLE_API_KEY
from lib.tool_registry import call_tool


DEFAULT_PCF_TABLE = "Transformation Projects"
DOCUMENTS_BASE_ID = os.getenv("DOCUMENTS_BASE_ID")
DOCUMENTS_TABLE_ID = os.getenv("DOCUMENTS_TABLE_ID")
PCF_BASE_ID = os.getenv("PCF_BASE_ID")
PCF_PARSER_API_KEY = os.getenv("AIR_TABLE_ACCESS_TOKEN")
PCF_PARSER_BASE_ID = os.getenv("AIR_TABLE_BASE_ID")
PCF_PARSER_TABLE_ID = os.getenv("AIR_TABLE_PCF_PARSER_TABLE_ID")

MAX_CONTEXT_FILES = 10
MAX_CONTEXT_FILE_CHARS = 2000

logger = logging.getLogger(__name__)


def _read_optional_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "r") as handle:
            return handle.read()
    except OSError:
        return ""


def _load_docs_context(base_dir: Optional[Path] = None) -> Dict[str, str]:
    base_dir = base_dir or Path(__file__).resolve().parent.parent
    docs_dir = base_dir / "docs"
    return {
        "coding_style": _read_optional_file(docs_dir / "CODING_STYLE.md"),
        "engineering_standard": _read_optional_file(docs_dir / "ENGINEERING_STANDARD.md"),
        "grading_rubric": _read_optional_file(docs_dir / "BETAFITS_CODE_GRADING_RUBRIC.md"),
        "langgraph_context": _read_optional_file(docs_dir / "LANGGRAPH_CONTEXT.md"),
    }


def _format_context_files_summary(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "None provided."
    lines = []
    for entry in entries:
        path = entry.get("path", "unknown")
        error = entry.get("error")
        if error:
            lines.append(f"- {path}: ERROR: {error}")
            continue
        truncated = entry.get("truncated", False)
        header = f"- {path}"
        if truncated:
            header = f"{header} (truncated)"
        lines.append(header)
        content = entry.get("content", "")
        if content:
            lines.append(content)
    return "\n".join(lines)


def _load_context_files(file_paths: Optional[List[str]]) -> Dict[str, Any]:
    if not file_paths:
        return {"extra_context_files": [], "extra_context_files_summary": "None provided."}
    entries: List[Dict[str, Any]] = []
    for path_str in file_paths[:MAX_CONTEXT_FILES]:
        entry: Dict[str, Any] = {"path": path_str}
        path = Path(path_str).expanduser()
        try:
            content = _read_context_file(path)
            truncated = False
            if len(content) > MAX_CONTEXT_FILE_CHARS:
                content = content[:MAX_CONTEXT_FILE_CHARS] + "\n... [truncated]"
                truncated = True
            entry["content"] = content
            entry["truncated"] = truncated
        except Exception as exc:
            entry["error"] = str(exc)
        entries.append(entry)
    return {
        "extra_context_files": entries,
        "extra_context_files_summary": _format_context_files_summary(entries),
    }


def _read_context_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX support requires python-docx") from exc
        doc = Document(str(path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return path.read_text(errors="replace")


def _split_libraries(libraries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    preferred_libraries: List[Dict[str, Any]] = []
    non_preferred_libraries: List[Dict[str, Any]] = []

    for library in libraries:
        fields = library.get("fields", {})
        is_preferred = False
        if "Preferred" in fields:
            is_preferred = fields.get("Preferred", False)
        elif "preferred" in fields:
            is_preferred = fields.get("preferred", False)
        elif "Status" in fields:
            status = fields.get("Status", "").lower()
            is_preferred = "preferred" in status or status == "preferred"

        if is_preferred:
            preferred_libraries.append(library)
        else:
            non_preferred_libraries.append(library)

    return {
        "libraries": libraries,
        "preferred_libraries": preferred_libraries,
        "non_preferred_libraries": non_preferred_libraries,
    }


def _filter_populate_mappings(field_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    populate_mappings = [
        mapping for mapping in field_mappings
        if mapping.get("fields", {}).get("Action Type") == "Populate"
    ]
    populate_mappings.sort(key=lambda x: x.get("fields", {}).get("Field Order", 999))
    return populate_mappings


def _load_pcf_context(
    pcf_record_id: Optional[str],
    pcf_table: str,
    api_key: Optional[str] = None,
    base_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not pcf_record_id:
        return {"pcf_context": None, "pcf_context_error": None}

    try:
        effective_api_key = api_key or AIRTABLE_API_KEY
        effective_base_id = base_id or PCF_BASE_ID
        if effective_base_id:
            api = Api(effective_api_key)
            base = api.base(effective_base_id)
            table = base.table(pcf_table)
            record = table.get(pcf_record_id)
        else:
            if api_key and api_key != AIRTABLE_API_KEY:
                raise RuntimeError("PCF base id is required when using a custom Airtable token.")
            record = get_record_by_id(pcf_table, pcf_record_id)
        return {
            "pcf_context": {
                "table": pcf_table,
                "record": {
                    "id": record.get("id"),
                    "fields": record.get("fields", {}),
                },
            },
            "pcf_context_error": None,
        }
    except Exception as exc:  # pragma: no cover - surface errors to caller
        return {
            "pcf_context": {
                "table": pcf_table,
                "record_id": pcf_record_id,
            },
            "pcf_context_error": str(exc),
        }


def _build_documents_text(doc: Optional[Dict[str, Any]]) -> str:
    if not doc:
        return ""
    notes = (doc.get("notes") or "").strip()
    url = (doc.get("url") or "").strip()
    attachments = doc.get("attachments") or []
    extra_lines = []
    if url:
        extra_lines.append(f"URL: {url}")
    if attachments:
        extra_lines.append("Attachments:")
        extra_lines.extend(attachments)
    if extra_lines:
        if notes:
            return f"{notes}\n\n" + "\n".join(extra_lines)
        return "\n".join(extra_lines)
    return notes


def _summarize_documents(docs: List[Dict[str, Any]], max_docs: int = 8, max_chars: int = 1200) -> str:
    if not docs:
        return "None provided."
    lines = []
    for doc in docs[:max_docs]:
        name = doc.get("name") or "Untitled"
        category = doc.get("category") or "Uncategorized"
        notes = doc.get("notes") or ""
        trimmed = notes.strip()
        if trimmed and len(trimmed) > max_chars:
            trimmed = f"{trimmed[:max_chars]}\n... [truncated]"
        line = f"- {name} ({category})"
        if trimmed:
            line = f"{line}: {trimmed}"
        lines.append(line)
        url = doc.get("url")
        if url:
            lines.append(f"  URL: {url}")
        attachments = doc.get("attachments") or []
        if attachments:
            lines.append("  Attachments:")
            lines.extend([f"  - {attachment}" for attachment in attachments[:3]])
    return "\n".join(lines)


def _fetch_documents_records(
    table,
    formula: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        if formula:
            return table.all(formula=formula)
        return table.all()
    except Exception as exc:  # pragma: no cover - surfaced to logs
        logger.warning("Documents/Resources fetch failed: %s", exc)
        return []


def _load_documents_context(pcf_record_id: Optional[str]) -> Dict[str, Any]:
    if not DOCUMENTS_BASE_ID or not DOCUMENTS_TABLE_ID:
        return {
            "documents_context_error": "Documents base/table not configured.",
            "standard_documents": [],
            "pcf_documents": [],
            "pcf_documents_summary": "None provided.",
            "coding_style": "",
            "engineering_standard": "",
        }

    api = Api(AIRTABLE_API_KEY)
    base = api.base(DOCUMENTS_BASE_ID)
    table = base.table(DOCUMENTS_TABLE_ID)

    name_formula = "OR({Name}='Engineering Standard', {Name}='Coding Style')"
    standard_records = _fetch_documents_records(table, name_formula)

    pcf_records: List[Dict[str, Any]] = []
    if pcf_record_id:
        pcf_formula = f'FIND("{pcf_record_id}", ARRAYJOIN({{Link to PCFs}}))'
        pcf_records = _fetch_documents_records(table, pcf_formula)

    def normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fields = record.get("fields", {})
        if fields.get("Hide"):
            return None
        attachments = fields.get("Attachments") or []
        attachment_urls = []
        for attachment in attachments:
            if isinstance(attachment, dict) and attachment.get("url"):
                attachment_urls.append(attachment["url"])
        return {
            "id": record.get("id"),
            "name": fields.get("Name", ""),
            "notes": fields.get("Notes", ""),
            "url": fields.get("URL", ""),
            "attachments": attachment_urls,
            "category": fields.get("Category", ""),
            "type": fields.get("Type", ""),
            "pcf_links": fields.get("Link to PCFs", []) or [],
        }

    standard_docs: List[Dict[str, Any]] = []
    for record in standard_records:
        doc = normalize_record(record)
        if doc:
            standard_docs.append(doc)

    pcf_docs: List[Dict[str, Any]] = []
    for record in pcf_records:
        doc = normalize_record(record)
        if doc:
            pcf_docs.append(doc)

    def select_doc(name_match: str) -> Optional[Dict[str, Any]]:
        candidates = [
            doc for doc in standard_docs
            if name_match.lower() in (doc.get("name") or "").lower()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda doc: len((doc.get("notes") or "").strip()))

    engineering_doc = select_doc("engineering standard")
    coding_doc = select_doc("coding style")

    return {
        "documents_context_error": None,
        "standard_documents": standard_docs,
        "pcf_documents": pcf_docs,
        "pcf_documents_summary": _summarize_documents(pcf_docs),
        "coding_style": _build_documents_text(coding_doc),
        "engineering_standard": _build_documents_text(engineering_doc),
    }


def build_generator_context(
    message: str,
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
    context_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build shared context for the code generator service."""
    repositories_fields = call_tool("get_repositories_fields")
    folders_fields = call_tool("get_folders_fields")
    files_fields = call_tool("get_files_fields")

    io_format_records = call_tool("get_code_generator_output_format")
    io_format = io_format_records[0] if io_format_records else {}

    field_mappings = call_tool("get_field_mappings_for_code_generator")
    populate_mappings = _filter_populate_mappings(field_mappings)

    libraries = call_tool("get_all_libraries")
    library_context = _split_libraries(libraries)

    context: Dict[str, Any] = {
        "message": message,
        "repositories_fields": repositories_fields,
        "folders_fields": folders_fields,
        "files_fields": files_fields,
        "io_format": io_format,
        "field_mappings": populate_mappings,
        **library_context,
    }

    docs_context = _load_docs_context()
    documents_context = _load_documents_context(pcf_record_id)
    if documents_context.get("coding_style"):
        docs_context["coding_style"] = documents_context["coding_style"]
    if documents_context.get("engineering_standard"):
        docs_context["engineering_standard"] = documents_context["engineering_standard"]
    docs_context["documents_context_error"] = documents_context.get("documents_context_error")
    docs_context["standard_documents"] = documents_context.get("standard_documents", [])
    docs_context["pcf_documents"] = documents_context.get("pcf_documents", [])
    docs_context["pcf_documents_summary"] = documents_context.get("pcf_documents_summary", "None provided.")

    context.update(docs_context)
    context.update(_load_pcf_context(pcf_record_id, pcf_table))
    context.update(_load_context_files(context_files))
    return context


def build_reviewer_context(
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
) -> Dict[str, Any]:
    """Build shared context for the code reviewer service."""
    repositories = call_tool("get_all_repositories")
    folders = call_tool("get_all_folders")
    files = call_tool("get_all_files")

    io_format_records = call_tool("get_code_reviewer_output_format")
    io_format = io_format_records[0] if io_format_records else {}

    field_mappings = call_tool("get_field_mappings_for_code_reviewer")
    populate_mappings = _filter_populate_mappings(field_mappings)

    libraries = call_tool("get_all_libraries")
    library_context = _split_libraries(libraries)

    context: Dict[str, Any] = {
        "repositories": repositories,
        "folders": folders,
        "files": files,
        "io_format": io_format,
        "field_mappings": populate_mappings,
        **library_context,
    }

    docs_context = _load_docs_context()
    documents_context = _load_documents_context(pcf_record_id)
    if documents_context.get("coding_style"):
        docs_context["coding_style"] = documents_context["coding_style"]
    if documents_context.get("engineering_standard"):
        docs_context["engineering_standard"] = documents_context["engineering_standard"]
    docs_context["documents_context_error"] = documents_context.get("documents_context_error")
    docs_context["standard_documents"] = documents_context.get("standard_documents", [])
    docs_context["pcf_documents"] = documents_context.get("pcf_documents", [])
    docs_context["pcf_documents_summary"] = documents_context.get("pcf_documents_summary", "None provided.")

    context.update(docs_context)
    context.update(_load_pcf_context(pcf_record_id, pcf_table))
    return context


def build_pcf_context(
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
) -> Dict[str, Any]:
    """Build PCF context (record + linked documents) for PCF parser workflows."""
    context: Dict[str, Any] = {}
    table_for_fetch = PCF_PARSER_TABLE_ID or pcf_table
    context.update(
        _load_pcf_context(
            pcf_record_id,
            table_for_fetch,
            api_key=PCF_PARSER_API_KEY,
            base_id=PCF_PARSER_BASE_ID,
        )
    )
    context.update(_load_documents_context(pcf_record_id))
    return context
