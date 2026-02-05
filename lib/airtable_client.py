"""Airtable client for fetching fields from repositories, folders, and files tables."""
from typing import List, Dict, Any
import logging
import os
from pyairtable import Api
from lib.config import AIRTABLE_API_KEY, AIRTABLE_BASE_ID

logger = logging.getLogger(__name__)

# Initialize pyairtable API client
api = Api(AIRTABLE_API_KEY)
base = api.base(AIRTABLE_BASE_ID)

TRANSFORMATION_PROJECTS_TABLE = os.getenv("TRANSFORMATION_PROJECTS_TABLE", "Transformation Projects")


def _safe_table_call(table, method_name: str, *args, **kwargs):
    """Invoke a pyairtable method and log failures with table context."""
    method = getattr(table, method_name)
    table_name = getattr(table, "name", getattr(table, "_table_name", "<unknown>"))
    try:
        return method(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - surfaces in logs
        logger.error("Airtable %s.%s failed: %s", table_name, method_name, exc)
        raise


def _get_table_fields(table_name: str) -> Dict[str, str]:
    """
    Get all fields from a table by fetching all records.
    
    Args:
        table_name: Name of the Airtable table
        
    Returns:
        Dictionary with field names and their types
    """
    table = base.table(table_name)
    
    # Fetch all records with pagination
    all_records = _safe_table_call(table, "all")
    
    if not all_records:
        return {}
    
    # Collect all unique fields across all records
    all_fields = {}
    for record in all_records:
        fields = record.get("fields", {})
        for field_name, field_value in fields.items():
            if field_name not in all_fields:
                all_fields[field_name] = []
            if field_value is not None:
                all_fields[field_name].append(field_value)
    
    # Infer types from field values across all records
    field_types = {}
    for field_name, values in all_fields.items():
        # Check all values to determine the most appropriate type
        types_found = set()
        for value in values:
            if isinstance(value, str):
                types_found.add("text")
            elif isinstance(value, int):
                types_found.add("number")
            elif isinstance(value, float):
                types_found.add("number")
            elif isinstance(value, bool):
                types_found.add("checkbox")
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    types_found.add("multipleSelects" if len(value) > 1 else "singleSelect")
                else:
                    types_found.add("array")
            else:
                types_found.add("unknown")
        
        # Determine the primary type
        if "text" in types_found:
            field_types[field_name] = "text"
        elif "number" in types_found:
            field_types[field_name] = "number"
        elif "checkbox" in types_found:
            field_types[field_name] = "checkbox"
        elif "multipleSelects" in types_found or "singleSelect" in types_found:
            # Check if it's typically multiple or single
            has_multiple = any(isinstance(v, list) and len(v) > 1 for v in values if isinstance(v, list))
            field_types[field_name] = "multipleSelects" if has_multiple else "singleSelect"
        elif "array" in types_found:
            field_types[field_name] = "array"
        else:
            field_types[field_name] = "unknown"
    
    return field_types


def get_repositories_fields() -> Dict[str, str]:
    """
    Get fields from the repositories table.
    
    Returns:
        Dictionary mapping field names to their types
    """
    return _get_table_fields("Repositories")


def get_all_repositories() -> List[Dict[str, Any]]:
    """
    Fetch all records from the Repositories table.

    This returns each repository with its Airtable record ID and all fields so
    downstream components (e.g., the code reviewer) can match user messages to
    specific repositories to review.

    Returns:
        List of repository records in the form:
        [
            {
                "id": "<record_id>",
                "fields": { ... },        # all Airtable fields
                "createdTime": "<iso>",
            },
            ...
        ]
    """
    table = base.table("Repositories")

    all_records = _safe_table_call(table, "all")

    result: List[Dict[str, Any]] = []
    for record in all_records:
        result.append(
            {
                "id": record.get("id"),
                "fields": record.get("fields", {}),
                "createdTime": record.get("createdTime"),
            }
        )

    return result


def get_all_folders() -> List[Dict[str, Any]]:
    """
    Fetch all records from the Folders table.

    Returns:
        List of folder records with id, fields, and createdTime.
    """
    table = base.table("Folders")
    all_records = _safe_table_call(table, "all")

    result: List[Dict[str, Any]] = []
    for record in all_records:
        result.append(
            {
                "id": record.get("id"),
                "fields": record.get("fields", {}),
                "createdTime": record.get("createdTime"),
            }
        )

    return result


def get_all_files() -> List[Dict[str, Any]]:
    """
    Fetch all records from the Files table.

    Returns:
        List of file records with id, fields, and createdTime.
    """
    table = base.table("Files")
    all_records = _safe_table_call(table, "all")

    result: List[Dict[str, Any]] = []
    for record in all_records:
        result.append(
            {
                "id": record.get("id"),
                "fields": record.get("fields", {}),
                "createdTime": record.get("createdTime"),
            }
        )

    return result


def get_folders_fields() -> Dict[str, str]:
    """
    Get fields from the folders table.
    
    Returns:
        Dictionary mapping field names to their types
    """
    return _get_table_fields("Folders")


def get_files_fields() -> Dict[str, str]:
    """
    Get fields from the files table.
    
    Returns:
        Dictionary mapping field names to their types
    """
    return _get_table_fields("Files")


def _fetch_record_by_id(table_name: str, record_id: str) -> Dict[str, Any]:
    """
    Fetch a single record by ID from a table.
    
    Args:
        table_name: Name of the Airtable table
        record_id: The record ID to fetch
        
    Returns:
        Record data with fields
    """
    table = base.table(table_name)
    return _safe_table_call(table, "get", record_id)


def get_record_by_id(table_name: str, record_id: str) -> Dict[str, Any]:
    """
    Fetch a single record by ID from a table.

    Args:
        table_name: Name of the Airtable table.
        record_id: Airtable record ID to fetch.

    Returns:
        Record data with fields.
    """
    return _fetch_record_by_id(table_name, record_id)


def get_code_generator_output_format() -> List[Dict[str, Any]]:
    """
    Fetch complete data from the Io Formats table where name is "Code Generator Output (Github Extractor)".
    Expands linked records to get actual data from related tables.
    
    Returns:
        List of records with all their fields and expanded linked record data
    """
    table = base.table("Io Formats")
    
    # Filter by name field
    filter_formula = "{Name} = 'Code Generator Output (Github Extractor)'"
    all_records = _safe_table_call(table, "all", formula=filter_formula)
    
    # Process records and expand linked records
    result = []
    for record in all_records:
        fields = record.get("fields", {}).copy()
        
        # Expand "Link to Field Mappings" if present
        if "Link to Field Mappings" in fields:
            field_mapping_ids = fields["Link to Field Mappings"]
            if isinstance(field_mapping_ids, list):
                expanded_mappings = []
                for mapping_id in field_mapping_ids:
                    try:
                        mapping_record = _fetch_record_by_id("Field Mappings", mapping_id)
                        expanded_mappings.append({
                            "id": mapping_record.get("id"),
                            "fields": mapping_record.get("fields", {}),
                        })
                    except Exception as e:
                        print(f"Warning: Could not fetch Field Mapping {mapping_id}: {e}")
                        expanded_mappings.append({"id": mapping_id, "error": str(e)})
                fields["Link to Field Mappings"] = expanded_mappings
        
        # Expand "Link to Rule Sets (from Field Mappings)" if present
        if "Link to Rule Sets (from Field Mappings)" in fields:
            rule_set_ids = fields["Link to Rule Sets (from Field Mappings)"]
            if isinstance(rule_set_ids, list):
                # Remove duplicates while preserving order
                unique_rule_set_ids = list(dict.fromkeys(rule_set_ids))
                expanded_rule_sets = []
                for rule_set_id in unique_rule_set_ids:
                    try:
                        rule_set_record = _fetch_record_by_id("Rule Sets", rule_set_id)
                        expanded_rule_sets.append({
                            "id": rule_set_record.get("id"),
                            "fields": rule_set_record.get("fields", {}),
                        })
                    except Exception as e:
                        print(f"Warning: Could not fetch Rule Set {rule_set_id}: {e}")
                        expanded_rule_sets.append({"id": rule_set_id, "error": str(e)})
                fields["Link to Rule Sets (from Field Mappings)"] = expanded_rule_sets
        
        # Expand "Transformation Projects" if present
        if "Transformation Projects" in fields:
            project_ids = fields["Transformation Projects"]
            if isinstance(project_ids, list):
                expanded_projects = []
                for project_id in project_ids:
                    try:
                        project_record = _fetch_record_by_id(TRANSFORMATION_PROJECTS_TABLE, project_id)
                        expanded_projects.append({
                            "id": project_record.get("id"),
                            "fields": project_record.get("fields", {}),
                        })
                    except Exception as e:
                        print(f"Warning: Could not fetch Transformation Project {project_id}: {e}")
                        expanded_projects.append({"id": project_id, "error": str(e)})
                fields["Transformation Projects"] = expanded_projects
        
        result.append({
            "id": record.get("id"),
            "fields": fields,
            "createdTime": record.get("createdTime"),
        })
    
    return result


def get_code_reviewer_output_format() -> List[Dict[str, Any]]:
    """
    Fetch data from the Io Formats table where name is "Code Reviewer Output".

    This defines the expected JSON / field structure for code reviewer output.

    Returns:
        List of records with all their fields (linked records are not expanded here).
    """
    table = base.table("Io Formats")

    # Filter by name field for the Code Reviewer output format
    filter_formula = "{Name} = 'Code Reviewer Output'"
    all_records = _safe_table_call(table, "all", formula=filter_formula)

    result: List[Dict[str, Any]] = []
    for record in all_records:
        result.append(
            {
                "id": record.get("id"),
                "fields": record.get("fields", {}),
                "createdTime": record.get("createdTime"),
            }
        )

    return result


def get_all_field_mappings(io_format_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch records from the Field Mappings table, optionally filtered by IO Format IDs.
    
    Args:
        io_format_ids: Optional list of IO Format record IDs to filter by.
                      If provided, only returns Field Mappings that have these IO Formats
                      in their "Link to IO Formats" field.
    
    Returns:
        List of Field Mapping records with all their fields.
        Linked record fields will contain record IDs (not expanded).
        Use _fetch_record_by_id() if you need to expand specific linked records.
    """
    table = base.table("Field Mappings")
    
    # Fetch all records (we'll filter client-side for linked records)
    all_records = _safe_table_call(table, "all")
    
    # Format results and filter by IO Format IDs if provided
    result = []
    for record in all_records:
        fields = record.get("fields", {})
        
        # Filter by IO Format IDs if provided
        if io_format_ids:
            link_to_io_formats = fields.get("Link to IO Formats", [])
            if isinstance(link_to_io_formats, list):
                # Check if any of the IO Format IDs are in the linked records
                if not any(io_format_id in link_to_io_formats for io_format_id in io_format_ids):
                    continue  # Skip this record if it doesn't match
            else:
                # If it's not a list, skip if filtering is required
                continue
        
        result.append({
            "id": record.get("id"),
            "fields": fields,
            "createdTime": record.get("createdTime"),
        })
    
    return result


def get_field_mappings_for_code_generator() -> List[Dict[str, Any]]:
    """
    Fetch Field Mappings filtered by the Code Generator Output IO Format.
    This is a convenience function that automatically gets the IO Format IDs
    from get_code_generator_output_format() and filters Field Mappings accordingly.
    
    Returns:
        List of Field Mapping records that are linked to the Code Generator Output IO Format.
    """
    # Get the IO Format records
    io_format_records = get_code_generator_output_format()
    
    # Extract IDs from the IO Format records
    io_format_ids = [record.get("id") for record in io_format_records if record.get("id")]
    
    if not io_format_ids:
        return []  # No IO Formats found, return empty list
    
    # Fetch Field Mappings filtered by these IO Format IDs
    return get_all_field_mappings(io_format_ids=io_format_ids)


def get_field_mappings_for_code_reviewer() -> List[Dict[str, Any]]:
    """
    Fetch Field Mappings filtered by the Code Reviewer Output IO Format.

    Convenience around get_all_field_mappings() for the code reviewer pipeline.

    Returns:
        List of Field Mapping records that are linked to the Code Reviewer Output IO Format.
    """
    io_format_records = get_code_reviewer_output_format()
    io_format_ids = [record.get("id") for record in io_format_records if record.get("id")]

    if not io_format_ids:
        return []

    return get_all_field_mappings(io_format_ids=io_format_ids)


def get_all_libraries() -> List[Dict[str, Any]]:
    """
    Fetch all records from the Libraries table.
    Libraries are categorized as preferred or non-preferred.
    
    Returns:
        List of all Library records with all their fields.
        Each record should have fields indicating if it's preferred or not.
    """
    table = base.table("Libraries")
    
    # Fetch all records
    all_records = _safe_table_call(table, "all")
    
    # Format results
    result = []
    for record in all_records:
        result.append({
            "id": record.get("id"),
            "fields": record.get("fields", {}),
            "createdTime": record.get("createdTime"),
        })
    
    return result


def _chunked(iterable: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    """Yield successive chunks from a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def batch_update_records(table_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Batch update Airtable records for the specified table.

    Args:
        table_name: Name of the Airtable table to update.
        records: List of {"id": "<record_id>", "fields": {...}} payloads.

    Returns:
        List of updated record responses from Airtable.
    """
    if not records:
        return []

    table = base.table(table_name)
    updated: List[Dict[str, Any]] = []

    for chunk in _chunked(records, 10):  # Airtable batch limit
        payload = []
        for record in chunk:
            record_id = record.get("id")
            fields = record.get("fields", {})
            if not record_id or not fields:
                continue
            payload.append({"id": record_id, "fields": fields})

        if not payload:
            continue

        updated.extend(_safe_table_call(table, "batch_update", payload))

    return updated


def batch_update_repositories(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch update repository records."""
    return batch_update_records("Repositories", records)


def batch_update_folders(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch update folder records."""
    return batch_update_records("Folders", records)


def batch_update_files(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch update file records."""
    return batch_update_records("Files", records)


def batch_create_records(table_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Batch create Airtable records for the specified table.

    Args:
        table_name: Name of the Airtable table to create records in.
        records: List of {"fields": {...}} payloads.

    Returns:
        List of created record responses from Airtable.
    """
    if not records:
        return []

    table = base.table(table_name)
    created: List[Dict[str, Any]] = []

    for chunk in _chunked(records, 10):  # Airtable batch limit
        payload = []
        for record in chunk:
            # Support callers that already provide raw field dicts
            if isinstance(record, dict) and "fields" not in record:
                fields = record
            else:
                fields = record.get("fields", {}) if isinstance(record, dict) else {}
            if not fields:
                continue
            payload.append(fields)

        if not payload:
            continue

        created.extend(_safe_table_call(table, "batch_create", payload))

    return created


def batch_create_repositories(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch create repository records."""
    return batch_create_records("Repositories", records)


def batch_create_folders(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch create folder records."""
    return batch_create_records("Folders", records)


def batch_create_files(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch create file records."""
    return batch_create_records("Files", records)
