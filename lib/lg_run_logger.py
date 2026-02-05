"""Logging helpers for LangGraph run records and state snapshots."""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Optional

from pyairtable import Api
from dotenv import load_dotenv

from lib.config import AIRTABLE_API_KEY

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

LG_RUNS_BASE_ID = os.getenv("LG_RUNS_BASE_ID")
LG_RUNS_TABLE_ID = os.getenv("LG_RUNS_TABLE_ID")
LG_SNAPSHOTS_TABLE_ID = os.getenv("LG_SNAPSHOTS_TABLE_ID")
LG_WORKFLOWS_TABLE_ID = os.getenv("LG_WORKFLOWS_TABLE_ID")
LG_NODES_TABLE_ID = os.getenv("LG_NODES_TABLE_ID")
LG_WORKFLOWS_NAME_FIELD = os.getenv("LG_WORKFLOWS_NAME_FIELD", "Name")
LG_NODES_NAME_FIELD = os.getenv("LG_NODES_NAME_FIELD", "Name")
LG_WORKFLOWS_NODES_FIELD = os.getenv("LG_WORKFLOWS_NODES_FIELD", "LangGraph Nodes")
LG_LOGGING_MODE = os.getenv("LG_LOGGING_MODE", "auto").lower()

LG_RUNS_FIELD_RUN_ID = os.getenv("LG_RUNS_FIELD_RUN_ID", "Run ID")
LG_RUNS_FIELD_WORKFLOW = os.getenv("LG_RUNS_FIELD_WORKFLOW", "Workflow")
LG_RUNS_FIELD_TRIGGERED_BY = os.getenv("LG_RUNS_FIELD_TRIGGERED_BY", "Triggered By")
LG_RUNS_FIELD_TRIGGER_SOURCE_ID = os.getenv("LG_RUNS_FIELD_TRIGGER_SOURCE_ID", "Trigger Source ID")
LG_RUNS_FIELD_STATUS = os.getenv("LG_RUNS_FIELD_STATUS", "Status")
LG_RUNS_FIELD_ENVIRONMENT = os.getenv("LG_RUNS_FIELD_ENVIRONMENT", "Environment")
LG_RUNS_FIELD_INPUT_SUMMARY = os.getenv("LG_RUNS_FIELD_INPUT_SUMMARY", "Input Summary")
LG_RUNS_FIELD_INPUT_PAYLOAD = os.getenv("LG_RUNS_FIELD_INPUT_PAYLOAD", "Input Payload (raw JSON)")
LG_RUNS_FIELD_OUTPUT_SUMMARY = os.getenv("LG_RUNS_FIELD_OUTPUT_SUMMARY", "Output Summary")
LG_RUNS_FIELD_OUTPUT_PAYLOAD = os.getenv("LG_RUNS_FIELD_OUTPUT_PAYLOAD", "Output Payload (raw JSON)")
LG_RUNS_FIELD_STEP_LOG = os.getenv("LG_RUNS_FIELD_STEP_LOG", "Step Log (compact JSON array)")
LG_RUNS_FIELD_ERROR_MESSAGE = os.getenv("LG_RUNS_FIELD_ERROR_MESSAGE", "Error Message")
LG_RUNS_FIELD_ERROR_PAYLOAD = os.getenv("LG_RUNS_FIELD_ERROR_PAYLOAD", "Error Payload (raw JSON)")
LG_RUNS_FIELD_ENDED_AT = os.getenv("LG_RUNS_FIELD_ENDED_AT", "Ended At")
LG_RUNS_FIELD_PCFS = os.getenv("LG_RUNS_FIELD_PCFS", "PCFs")

LG_SNAPSHOTS_FIELD_SNAPSHOT_ID = os.getenv("LG_SNAPSHOTS_FIELD_SNAPSHOT_ID", "Snapshot ID")
LG_SNAPSHOTS_FIELD_RUN = os.getenv("LG_SNAPSHOTS_FIELD_RUN", "Run")
LG_SNAPSHOTS_FIELD_NODE = os.getenv("LG_SNAPSHOTS_FIELD_NODE", "Node")
LG_SNAPSHOTS_FIELD_INDEX = os.getenv("LG_SNAPSHOTS_FIELD_INDEX", "Snapshot Index")
LG_SNAPSHOTS_FIELD_TYPE = os.getenv("LG_SNAPSHOTS_FIELD_TYPE", "Snapshot Type")
LG_SNAPSHOTS_FIELD_STATE_JSON = os.getenv("LG_SNAPSHOTS_FIELD_STATE_JSON", "State Envelope JSON")
LG_SNAPSHOTS_FIELD_SCHEMA_REF = os.getenv("LG_SNAPSHOTS_FIELD_SCHEMA_REF", "Payload Schema Ref")
LG_SNAPSHOTS_FIELD_SCHEMA_VERSION = os.getenv("LG_SNAPSHOTS_FIELD_SCHEMA_VERSION", "Payload Version")
LG_SNAPSHOTS_FIELD_STATE_HASH = os.getenv("LG_SNAPSHOTS_FIELD_STATE_HASH", "State Hash")

LG_RUNS_MIN_FIELD_NAME = os.getenv("LG_RUNS_MIN_FIELD_NAME", "Name")
LG_RUNS_MIN_FIELD_DESCRIPTION = os.getenv("LG_RUNS_MIN_FIELD_DESCRIPTION", "Description")
LG_RUNS_MIN_FIELD_STATUS = os.getenv("LG_RUNS_MIN_FIELD_STATUS", "Status")
LG_RUNS_MIN_FIELD_START_DATE = os.getenv("LG_RUNS_MIN_FIELD_START_DATE", "Start Date")
LG_RUNS_MIN_FIELD_END_DATE = os.getenv("LG_RUNS_MIN_FIELD_END_DATE", "End Date")

LG_SNAPSHOTS_MIN_FIELD_TITLE = os.getenv("LG_SNAPSHOTS_MIN_FIELD_TITLE", "Title")
LG_SNAPSHOTS_MIN_FIELD_DETAIL = os.getenv("LG_SNAPSHOTS_MIN_FIELD_DETAIL", "Detail")
LG_SNAPSHOTS_MIN_FIELD_CREATED_DATE = os.getenv("LG_SNAPSHOTS_MIN_FIELD_CREATED_DATE", "Created Date")
LG_SNAPSHOTS_MIN_FIELD_RUN_LINK = os.getenv(
    "LG_SNAPSHOTS_MIN_FIELD_RUN_LINK", "Related LG Runs Entries"
)

_workflow_id_cache: Dict[str, str] = {}
_node_id_cache: Dict[str, str] = {}


def is_logging_configured() -> bool:
    return bool(LG_RUNS_BASE_ID and LG_RUNS_TABLE_ID and LG_SNAPSHOTS_TABLE_ID)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _base(base_id: str):
    api = Api(AIRTABLE_API_KEY)
    return api.base(base_id)


def _table(base_id: str, table_id: str):
    return _base(base_id).table(table_id)


def _generate_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"LG-RUN-{stamp}-{suffix}"


def _generate_snapshot_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"LG-SNAP-{stamp}-{suffix}"


def _read_env_value(key: str) -> Optional[str]:
    if not _env_path.exists():
        return None
    try:
        with open(_env_path, "r") as handle:
            for line in handle:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                env_key, env_value = raw.split("=", 1)
                if env_key.strip() == key:
                    return env_value.strip()
    except OSError:
        return None
    return None


def _is_unknown_field_error(exc: Exception) -> bool:
    return "UNKNOWN_FIELD_NAME" in str(exc)


def _is_invalid_value_error(exc: Exception) -> bool:
    return "INVALID_VALUE_FOR_COLUMN" in str(exc)


def _minimal_status(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"completed", "success"}:
        return "Completed"
    if normalized in {"failed", "error"}:
        return "Completed"
    if normalized in {"cancelled", "canceled"}:
        return "Archived"
    if normalized in {"running", "in progress"}:
        return "In Progress"
    return "In Progress"


def _build_run_fields_full(
    run_id: str,
    workflow_id: Optional[str],
    triggered_by: str,
    trigger_source_id: Optional[str],
    environment: str,
    input_summary: str,
    input_payload: Dict[str, Any],
    pcf_record_ids: Optional[list[str]],
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        LG_RUNS_FIELD_RUN_ID: run_id,
        LG_RUNS_FIELD_TRIGGERED_BY: triggered_by,
        LG_RUNS_FIELD_STATUS: "Running",
        LG_RUNS_FIELD_ENVIRONMENT: environment,
        LG_RUNS_FIELD_INPUT_SUMMARY: input_summary,
        LG_RUNS_FIELD_INPUT_PAYLOAD: json.dumps(input_payload, default=str),
    }
    if trigger_source_id:
        fields[LG_RUNS_FIELD_TRIGGER_SOURCE_ID] = trigger_source_id
    if workflow_id:
        fields[LG_RUNS_FIELD_WORKFLOW] = [workflow_id]
    if pcf_record_ids:
        fields[LG_RUNS_FIELD_PCFS] = pcf_record_ids
    return fields


def _build_run_fields_minimal(
    run_id: str,
    input_summary: str,
) -> Dict[str, Any]:
    return {
        LG_RUNS_MIN_FIELD_NAME: run_id,
        LG_RUNS_MIN_FIELD_DESCRIPTION: input_summary,
        LG_RUNS_MIN_FIELD_STATUS: "In Progress",
        LG_RUNS_MIN_FIELD_START_DATE: _utc_date(),
    }


def _resolve_record_id(
    base_id: str,
    table_id: str,
    name_field: str,
    name_value: str,
    cache: Dict[str, str],
) -> Optional[str]:
    if not name_value or not table_id:
        return None
    if name_value in cache:
        return cache[name_value]
    table = _table(base_id, table_id)
    formula = f"{{{name_field}}}='{name_value}'"
    try:
        records = table.all(formula=formula)
    except Exception as exc:  # pragma: no cover - surfaced in logs
        logger.warning("LG logging lookup failed for %s: %s", name_value, exc)
        return None
    if not records:
        return None
    record_id = records[0].get("id")
    if record_id:
        cache[name_value] = record_id
    return record_id


def _normalize_name(value: str) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def _fetch_node_name_map() -> Dict[str, str]:
    if not LG_RUNS_BASE_ID or not LG_NODES_TABLE_ID:
        return {}
    table = _table(LG_RUNS_BASE_ID, LG_NODES_TABLE_ID)
    try:
        records = table.all()
    except Exception as exc:  # pragma: no cover - surfaced in logs
        logger.warning("LG logging node fetch failed: %s", exc)
        return {}
    node_map: Dict[str, str] = {}
    for record in records:
        fields = record.get("fields", {})
        name = fields.get(LG_NODES_NAME_FIELD)
        if name:
            node_map[record.get("id")] = _normalize_name(name)
    return node_map


def resolve_workflow_id_by_nodes(node_names: list[str]) -> Optional[str]:
    if not LG_RUNS_BASE_ID or not LG_WORKFLOWS_TABLE_ID:
        return None
    desired_nodes = {_normalize_name(name) for name in node_names if name}
    if not desired_nodes:
        return None

    table = _table(LG_RUNS_BASE_ID, LG_WORKFLOWS_TABLE_ID)
    try:
        workflows = table.all()
    except Exception as exc:  # pragma: no cover - surfaced in logs
        logger.warning("LG logging workflow fetch failed: %s", exc)
        return None

    node_map = _fetch_node_name_map()
    best_record_id: Optional[str] = None
    best_score = -1
    best_ratio = -1.0

    for record in workflows:
        fields = record.get("fields", {})
        linked_nodes = fields.get(LG_WORKFLOWS_NODES_FIELD, []) or []
        linked_names = {node_map.get(node_id) for node_id in linked_nodes}
        linked_names.discard(None)
        score = len(desired_nodes & linked_names)
        ratio = score / max(len(desired_nodes), 1)
        if score > best_score or (score == best_score and ratio > best_ratio):
            best_score = score
            best_ratio = ratio
            best_record_id = record.get("id")

    if best_score <= 0:
        if len(workflows) == 1:
            return workflows[0].get("id")
        logger.warning("LG logging: no workflow match found for nodes %s", sorted(desired_nodes))
        return None

    return best_record_id


def resolve_workflow_id(workflow_name: Optional[str]) -> Optional[str]:
    if not workflow_name or not LG_WORKFLOWS_TABLE_ID or not LG_RUNS_BASE_ID:
        return None
    return _resolve_record_id(
        LG_RUNS_BASE_ID,
        LG_WORKFLOWS_TABLE_ID,
        LG_WORKFLOWS_NAME_FIELD,
        workflow_name,
        _workflow_id_cache,
    )


def resolve_node_id(node_name: Optional[str]) -> Optional[str]:
    if not node_name or not LG_NODES_TABLE_ID or not LG_RUNS_BASE_ID:
        return None
    resolved = _resolve_record_id(
        LG_RUNS_BASE_ID,
        LG_NODES_TABLE_ID,
        LG_NODES_NAME_FIELD,
        node_name,
        _node_id_cache,
    )
    if resolved:
        return resolved

    table = _table(LG_RUNS_BASE_ID, LG_NODES_TABLE_ID)
    target = _normalize_name(node_name)
    try:
        records = table.all()
    except Exception as exc:  # pragma: no cover - surfaced in logs
        logger.warning("LG logging node fetch failed: %s", exc)
        return None
    for record in records:
        fields = record.get("fields", {})
        name = fields.get(LG_NODES_NAME_FIELD)
        if name and _normalize_name(name) == target:
            record_id = record.get("id")
            if record_id:
                _node_id_cache[node_name] = record_id
                return record_id
    return None


def create_run(
    workflow_name: Optional[str],
    workflow_id: Optional[str],
    node_names: Optional[list[str]],
    triggered_by: str,
    trigger_source_id: Optional[str],
    environment: str,
    input_summary: str,
    input_payload: Dict[str, Any],
    pcf_record_ids: Optional[list[str]] = None,
) -> Dict[str, Any]:
    if not is_logging_configured():
        return {}

    run_id = _generate_run_id()
    resolved_workflow_id = workflow_id or resolve_workflow_id(workflow_name)
    if not resolved_workflow_id and node_names:
        resolved_workflow_id = resolve_workflow_id_by_nodes(node_names)
    if not resolved_workflow_id:
        logger.warning("LG logging: workflow id not resolved; skipping run creation.")
        return {}
    table = _table(LG_RUNS_BASE_ID, LG_RUNS_TABLE_ID)
    mode = (os.getenv("LG_LOGGING_MODE", LG_LOGGING_MODE) or "auto").strip().lower()
    if mode == "auto":
        env_mode = _read_env_value("LG_LOGGING_MODE")
        if env_mode:
            mode = env_mode.strip().lower()
    record = None
    logging_mode = "full"

    if mode == "minimal":
        fields = _build_run_fields_minimal(run_id, input_summary)
        try:
            record = table.create(fields)
            logging_mode = "minimal"
        except Exception as exc:  # pragma: no cover - surfaced in logs
            if _is_invalid_value_error(exc):
                logger.warning("LG logging: retrying minimal run without dates: %s", exc)
                fields.pop(LG_RUNS_MIN_FIELD_START_DATE, None)
                try:
                    record = table.create(fields)
                    logging_mode = "minimal"
                except Exception as retry_exc:
                    if _is_invalid_value_error(retry_exc):
                        logger.warning(
                            "LG logging: retrying minimal run without status: %s",
                            retry_exc,
                        )
                        fields.pop(LG_RUNS_MIN_FIELD_STATUS, None)
                        try:
                            record = table.create(fields)
                            logging_mode = "minimal"
                        except Exception as last_exc:
                            logger.warning(
                                "LG logging: failed to create minimal run record: %s",
                                last_exc,
                            )
                            return {}
                    else:
                        logger.warning("LG logging: failed to create minimal run record: %s", retry_exc)
                        return {}
            else:
                logger.warning("LG logging: failed to create minimal run record: %s", exc)
                return {}
    else:
        fields = _build_run_fields_full(
            run_id,
            resolved_workflow_id,
            triggered_by,
            trigger_source_id,
            environment,
            input_summary,
            input_payload,
            pcf_record_ids,
        )
        try:
            record = table.create(fields)
        except Exception as exc:  # pragma: no cover - surfaced in logs
            if mode == "auto" and _is_unknown_field_error(exc):
                logger.warning("LG logging: falling back to minimal run fields: %s", exc)
                try:
                    minimal_fields = _build_run_fields_minimal(run_id, input_summary)
                    record = table.create(minimal_fields)
                    logging_mode = "minimal"
                except Exception as fallback_exc:
                    if _is_invalid_value_error(fallback_exc):
                        logger.warning(
                            "LG logging: retrying minimal run without dates: %s",
                            fallback_exc,
                        )
                        minimal_fields = _build_run_fields_minimal(run_id, input_summary)
                        minimal_fields.pop(LG_RUNS_MIN_FIELD_START_DATE, None)
                        try:
                            record = table.create(minimal_fields)
                            logging_mode = "minimal"
                        except Exception as retry_exc:
                            if _is_invalid_value_error(retry_exc):
                                logger.warning(
                                    "LG logging: retrying minimal run without status: %s",
                                    retry_exc,
                                )
                                minimal_fields.pop(LG_RUNS_MIN_FIELD_STATUS, None)
                                try:
                                    record = table.create(minimal_fields)
                                    logging_mode = "minimal"
                                except Exception as last_exc:
                                    logger.warning(
                                        "LG logging: failed to create minimal run record: %s",
                                        last_exc,
                                    )
                                    return {}
                            else:
                                logger.warning(
                                    "LG logging: failed to create minimal run record: %s",
                                    retry_exc,
                                )
                                return {}
                    else:
                        logger.warning("LG logging: failed to create minimal run record: %s", fallback_exc)
                        return {}
            else:
                logger.warning("LG logging: failed to create run record: %s", exc)
                return {}
    return {
        "run_record_id": record.get("id"),
        "run_id": run_id,
        "workflow_id": resolved_workflow_id,
        "workflow_name": workflow_name,
        "snapshot_index": 0,
        "step_log": [],
        "logging_mode": logging_mode,
    }


def log_snapshot(
    state: Dict[str, Any],
    node_name: str,
    snapshot_type: str,
) -> None:
    if not is_logging_configured():
        return

    run_context = state.get("_run_context")
    if not run_context:
        return

    run_record_id = run_context.get("run_record_id")
    if not run_record_id:
        return

    snapshot_index = int(run_context.get("snapshot_index", 0))
    payload_json = json.dumps(state, default=str)
    snapshot_id = _generate_snapshot_id()
    logging_mode = run_context.get("logging_mode", "full")

    table = _table(LG_RUNS_BASE_ID, LG_SNAPSHOTS_TABLE_ID)

    if logging_mode == "minimal":
        fields: Dict[str, Any] = {
            LG_SNAPSHOTS_MIN_FIELD_TITLE: f"{node_name}-{snapshot_type}",
            LG_SNAPSHOTS_MIN_FIELD_DETAIL: payload_json,
            LG_SNAPSHOTS_MIN_FIELD_CREATED_DATE: _utc_date(),
            LG_SNAPSHOTS_MIN_FIELD_RUN_LINK: [run_record_id],
        }
        try:
            table.create(fields)
        except Exception as exc:  # pragma: no cover - surfaced in logs
            if _is_invalid_value_error(exc):
                logger.warning(
                    "LG logging: retrying minimal snapshot without created date: %s",
                    exc,
                )
                fields.pop(LG_SNAPSHOTS_MIN_FIELD_CREATED_DATE, None)
                try:
                    table.create(fields)
                except Exception as retry_exc:
                    logger.warning(
                        "LG logging: failed to create minimal snapshot record: %s",
                        retry_exc,
                    )
                    return
            else:
                logger.warning("LG logging: failed to create minimal snapshot record: %s", exc)
                return
    else:
        node_id = resolve_node_id(node_name)
        if not node_id:
            logger.warning("LG logging: node id not found for %s", node_name)
            return
        state_hash = sha256(payload_json.encode("utf-8")).hexdigest()
        fields = {
            LG_SNAPSHOTS_FIELD_SNAPSHOT_ID: snapshot_id,
            LG_SNAPSHOTS_FIELD_RUN: [run_record_id],
            LG_SNAPSHOTS_FIELD_NODE: [node_id],
            LG_SNAPSHOTS_FIELD_INDEX: snapshot_index,
            LG_SNAPSHOTS_FIELD_TYPE: snapshot_type,
            LG_SNAPSHOTS_FIELD_STATE_JSON: payload_json,
            LG_SNAPSHOTS_FIELD_STATE_HASH: state_hash,
        }
        try:
            table.create(fields)
        except Exception as exc:  # pragma: no cover - surfaced in logs
            if LG_LOGGING_MODE == "auto" and _is_unknown_field_error(exc):
                logger.warning("LG logging: falling back to minimal snapshot fields: %s", exc)
                try:
                    table.create(
                        {
                            LG_SNAPSHOTS_MIN_FIELD_TITLE: f"{node_name}-{snapshot_type}",
                            LG_SNAPSHOTS_MIN_FIELD_DETAIL: payload_json,
                            LG_SNAPSHOTS_MIN_FIELD_CREATED_DATE: _utc_date(),
                            LG_SNAPSHOTS_MIN_FIELD_RUN_LINK: [run_record_id],
                        }
                    )
                    run_context["logging_mode"] = "minimal"
                except Exception as fallback_exc:
                    logger.warning(
                        "LG logging: failed to create minimal snapshot record: %s",
                        fallback_exc,
                    )
                    return
            else:
                logger.warning("LG logging: failed to create snapshot record: %s", exc)
                return

    step_entry = {
        "index": snapshot_index,
        "node": node_name,
        "type": snapshot_type,
        "timestamp": _utc_now(),
    }
    run_context["snapshot_index"] = snapshot_index + 1
    run_context.setdefault("step_log", []).append(step_entry)
    state["_run_context"] = run_context


def summarize_payload(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    summary = payload.get("summary")
    if isinstance(summary, dict):
        name = summary.get("repository_name")
        totals = [
            summary.get("total_repositories"),
            summary.get("total_folders"),
            summary.get("total_files"),
        ]
        if name:
            return f"Repository: {name} (repos/folders/files: {totals})"
    if "error" in payload:
        return f"Error: {payload.get('error')}"
    return json.dumps({k: payload.get(k) for k in list(payload.keys())[:5]}, default=str)


def finalize_run(
    run_context: Dict[str, Any],
    status: str,
    output_payload: Optional[Dict[str, Any]] = None,
    output_summary: Optional[str] = None,
    error_message: Optional[str] = None,
    error_payload: Optional[Any] = None,
) -> None:
    if not is_logging_configured() or not run_context:
        return

    run_record_id = run_context.get("run_record_id")
    if not run_record_id:
        return

    logging_mode = run_context.get("logging_mode", "full")
    table = _table(LG_RUNS_BASE_ID, LG_RUNS_TABLE_ID)

    if logging_mode == "minimal":
        description_parts = []
        if output_summary:
            description_parts.append(f"Output: {output_summary}")
        if error_message:
            description_parts.append(f"Error: {error_message}")
        description_text = "\n".join(description_parts) if description_parts else None
        fields = {
            LG_RUNS_MIN_FIELD_STATUS: _minimal_status(status),
            LG_RUNS_MIN_FIELD_END_DATE: _utc_date(),
        }
        if description_text:
            fields[LG_RUNS_MIN_FIELD_DESCRIPTION] = description_text
        try:
            table.update(run_record_id, fields)
        except Exception as exc:  # pragma: no cover - surfaced in logs
            if _is_invalid_value_error(exc):
                logger.warning(
                    "LG logging: retrying minimal finalize without end date: %s",
                    exc,
                )
                fields.pop(LG_RUNS_MIN_FIELD_END_DATE, None)
                try:
                    table.update(run_record_id, fields)
                except Exception as retry_exc:
                    if _is_invalid_value_error(retry_exc):
                        logger.warning(
                            "LG logging: retrying minimal finalize without status: %s",
                            retry_exc,
                        )
                        fields.pop(LG_RUNS_MIN_FIELD_STATUS, None)
                        try:
                            table.update(run_record_id, fields)
                        except Exception as last_exc:
                            logger.warning(
                                "LG logging: failed to finalize minimal run record: %s",
                                last_exc,
                            )
                    else:
                        logger.warning(
                            "LG logging: failed to finalize minimal run record: %s",
                            retry_exc,
                        )
            else:
                logger.warning("LG logging: failed to finalize minimal run record: %s", exc)
        return

    fields = {
        LG_RUNS_FIELD_STATUS: status,
        LG_RUNS_FIELD_ENDED_AT: _utc_now(),
    }
    if output_payload is not None:
        fields[LG_RUNS_FIELD_OUTPUT_PAYLOAD] = json.dumps(output_payload, default=str)
    if output_summary:
        fields[LG_RUNS_FIELD_OUTPUT_SUMMARY] = output_summary
    if error_message:
        fields[LG_RUNS_FIELD_ERROR_MESSAGE] = error_message
    if error_payload is not None:
        fields[LG_RUNS_FIELD_ERROR_PAYLOAD] = json.dumps(error_payload, default=str)
    step_log = run_context.get("step_log")
    if step_log is not None:
        fields[LG_RUNS_FIELD_STEP_LOG] = json.dumps(step_log, default=str)

    try:
        table.update(run_record_id, fields)
    except Exception as exc:  # pragma: no cover - surfaced in logs
        if LG_LOGGING_MODE == "auto" and _is_unknown_field_error(exc):
            logger.warning("LG logging: falling back to minimal run finalize: %s", exc)
            run_context["logging_mode"] = "minimal"
            finalize_run(
                run_context,
                status=status,
                output_payload=output_payload,
                output_summary=output_summary,
                error_message=error_message,
                error_payload=error_payload,
            )
        else:
            logger.warning("LG logging: failed to finalize run record: %s", exc)
