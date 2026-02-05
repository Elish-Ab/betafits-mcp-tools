"""CLI harness for invoking the LangGraph workflow end-to-end."""
import argparse
import json
import os
import sys
from pathlib import Path

from lib.validation import validate_user_message
from workflows.langgraph.orchestrator.graph import workflow
from services.code_generator.run import run as run_generator_service
from lib.context_engine import DEFAULT_PCF_TABLE
from lib.lg_run_logger import (
    create_run,
    finalize_run,
    is_logging_configured,
    summarize_payload,
)


def should_save_output() -> bool:
    """Determine whether to save workflow output to disk."""
    flag = os.getenv("WORKFLOW_SAVE_OUTPUT", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def test_workflow(
    message: str,
    output_file: str = "workflow_output.json",
    save_output: bool = True,
    run_context: dict | None = None,
    context_files: list[str] | None = None,
):
    """
    Test the LangGraph workflow with a message.
    
    Args:
        message: The input message to process
        output_file: Name of the output JSON file
    """
    print("=" * 60)
    print("Testing LangGraph Workflow")
    print("=" * 60)
    print(f"\nInput Message: {message}\n")
    
    try:
        # Prepare initial state
        initial_state = {
            "message": message,
            "next_node": None,
            "code_generator_output": None,
            "pcf_parser_output": None,
            "error": None,
            "pcf_record_id": args.pcf_record_id,
            "pcf_table": args.pcf_table,
            "meeting_record_id": None,
            "transcript": None,
        }
        if run_context:
            initial_state["_run_context"] = run_context
        if context_files:
            initial_state["context_files"] = context_files
        
        print("Running workflow...")
        print("-" * 60)
        
        # Invoke the workflow
        result = workflow.invoke(initial_state)
        
        print("\nWorkflow completed!")
        print("-" * 60)
        
        # Display results
        print("\nWorkflow State:")
        print(f"  Message: {result.get('message', '')}")
        print(f"  Next Node: {result.get('next_node', 'None')}")
        print(f"  Error: {result.get('error', 'None')}")
        
        # Check for code generator output
        code_output = result.get("code_generator_output")
        if code_output:
            print("\n✓ Code Generator Output Found!")
            print(f"  Output keys: {list(code_output.keys())}")
            
            # Check for repository structure (matching Airtable schema)
            if "repository" in code_output:
                repository_structure = code_output["repository"]
                repos = repository_structure.get("repositories", [])
                folders = repository_structure.get("folders", [])
                files = repository_structure.get("files", [])
                files_with_code = [f for f in files if f.get("Source Code")]
                
                repo_name = repos[0].get("Name", "unknown") if repos else "unknown"
                
                print(f"\n  Repository Structure (Airtable Schema):")
                print(f"    Repositories: {len(repos)}")
                print(f"      Name: {repo_name}")
                print(f"    Folders: {len(folders)}")
                print(f"    Files: {len(files)}")
                print(f"    Files with code: {len(files_with_code)}")
                if files_with_code:
                    file_names = [f.get("File Name", f.get("Relative File Path", "")) for f in files_with_code[:5]]
                    print(f"      Files: {file_names}{'...' if len(files_with_code) > 5 else ''}")
                
                # Show summary
                summary = code_output.get("summary", {})
                if summary:
                    print(f"\n  Summary:")
                    print(f"    Repository name: {summary.get('repository_name', 'N/A')}")
                    print(f"    Total repositories: {summary.get('total_repositories', 0)}")
                    print(f"    Total folders: {summary.get('total_folders', 0)}")
                    print(f"    Total files: {summary.get('total_files', 0)}")
                    print(f"    Files with code: {summary.get('files_with_code', 0)}")
                
                # Repository is stored in JSON only (matching Airtable schema, ready to push)
                print(f"\n  Repository stored in JSON structure matching Airtable schema (ready to push)")
            elif "output" in code_output:
                # Fallback for old format
                output_data = code_output["output"]
                code_files = output_data.get("code_files", {})
                json_files = output_data.get("json_files", {})
                print(f"  Code files generated: {len(code_files)}")
                print(f"  JSON files generated: {len(json_files)}")
        else:
            print("\n⚠ No code generator output found")

        pcf_output = result.get("pcf_parser_output")
        if pcf_output:
            print("\n✓ PCF Parser Output Found!")
            print(f"  Output keys: {list(pcf_output.keys())}")
        
        if save_output:
            # Save only the repository JSON structure (single output file)
            if code_output and "repository" in code_output:
                repository_structure = code_output.get("repository", {})
                repos = repository_structure.get("repositories", [])
                repo_name = repos[0].get("Name", "generated_repository") if repos else "generated_repository"
                
                # Use the repository name for the output file, or default to output_file
                if repo_name and repo_name != "unknown":
                    # Sanitize repo name for filename (kebab-case, remove special chars)
                    import re
                    safe_name = re.sub(r'[^a-zA-Z0-9-]', '-', repo_name.lower())
                    safe_name = re.sub(r'-+', '-', safe_name).strip('-')
                    output_file = f"{safe_name}.json"
                else:
                    output_file = "repository_output.json"
                
                output_path = Path(output_file)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(repository_structure, f, indent=2, default=str, ensure_ascii=False)
                
                print(f"\n✓ Repository JSON saved to: {output_path.absolute()}")
                print(f"  Repository name: {repo_name}")
                print(f"  Contains {len(repository_structure.get('repositories', []))} repository(ies)")
                print(f"  Contains {len(repository_structure.get('folders', []))} folder(s)")
                print(f"  Contains {len(repository_structure.get('files', []))} file(s)")
                print(f"  Structure matches Airtable schema (ready to push)")
            else:
                # Fallback: save full workflow result if no repository structure
                output_path = Path(output_file)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\n✓ Results saved to: {output_path.absolute()}")
        else:
            print("\nℹ Output file write skipped (WORKFLOW_SAVE_OUTPUT disabled).")
        
        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the LangGraph workflow once.")
    parser.add_argument(
        "message",
        nargs="*",
        default=["Generate a folder structure for a Python project with a main.py file and a requirements.txt file"],
        help="Message to send into the workflow.",
    )
    parser.add_argument(
        "--output",
        default="workflow_output.json",
        help="Path to write workflow output JSON when persistence is enabled.",
    )
    parser.add_argument(
        "--repo-name",
        help="Name of an existing repository to enhance (bypasses LangGraph).",
    )
    parser.add_argument(
        "--service-mode",
        action="store_true",
        help="Call the code generator service directly instead of LangGraph. Automatically enabled when --repo-name is provided.",
    )
    parser.add_argument(
        "--pcf-record-id",
        help="Optional Airtable record ID for PCF context (shared across tools).",
    )
    parser.add_argument(
        "--pcf-table",
        default=DEFAULT_PCF_TABLE,
        help="Airtable table name containing the PCF record context.",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="Optional file path to inject as additional context (can be used multiple times).",
    )
    parser.add_argument(
        "--log-runs",
        action="store_true",
        help="Persist run logs and snapshots to Airtable (LG Runs + LG State Snapshots).",
    )
    parser.add_argument(
        "--workflow-name",
        default=None,
        help="Workflow name used for LG Runs logging (resolved via LG Workflows table).",
    )
    parser.add_argument(
        "--workflow-id",
        help="Explicit workflow record ID for LG Runs logging (bypasses name lookup).",
    )
    parser.add_argument(
        "--triggered-by",
        default="Manual",
        help="Trigger source for LG Runs logging (Slack/Schedule/API/Manual/Internal Event).",
    )
    parser.add_argument(
        "--trigger-source-id",
        help="Trigger source reference (Slack ts / cron id / webhook id).",
    )
    parser.add_argument(
        "--environment",
        default=os.getenv("ENVIRONMENT", "Local"),
        help="Environment value for LG Runs logging.",
    )
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    """Entry point used by `python -m apps.workflow_cli.main`."""
    args = _parse_args(argv)
    message = validate_user_message(" ".join(args.message))

    print(f"\nUsing message: {message}\n")

    run_context = {}
    should_log_runs = args.log_runs and is_logging_configured()
    if should_log_runs:
        if args.repo_name or args.service_mode:
            node_names_for_lookup = ["code_generator"]
        else:
            node_names_for_lookup = ["mcp_brain", "code_generator", "code_reviewer", "pcf_parser"]
        run_context = create_run(
            workflow_name=args.workflow_name,
            workflow_id=args.workflow_id,
            node_names=node_names_for_lookup,
            triggered_by=args.triggered_by,
            trigger_source_id=args.trigger_source_id,
            environment=args.environment,
            input_summary=message,
            input_payload={
                "message": message,
                "repo_name": args.repo_name,
                "pcf_record_id": args.pcf_record_id,
            },
            pcf_record_ids=[args.pcf_record_id] if args.pcf_record_id else None,
        )
        if not run_context:
            should_log_runs = False

    if args.repo_name:
        print("Enhancement mode enabled; invoking code generator service directly.")
        result = run_generator_service(
            message,
            use_chain=True,
            persist=True,
            repo_name=args.repo_name,
            pcf_record_id=args.pcf_record_id,
            pcf_table=args.pcf_table,
            context_files=args.context_file,
        )
        print(json.dumps(result, indent=2, default=str))
        if should_log_runs:
            finalize_run(
                run_context,
                status="Completed",
                output_payload=result,
                output_summary=summarize_payload(result),
            )
        return 0

    if args.service_mode:
        print("Direct service mode requested; bypassing LangGraph.")
        result = run_generator_service(
            message,
            use_chain=True,
            persist=True,
            pcf_record_id=args.pcf_record_id,
            pcf_table=args.pcf_table,
            context_files=args.context_file,
        )
        print(json.dumps(result, indent=2, default=str))
        if should_log_runs:
            finalize_run(
                run_context,
                status="Completed",
                output_payload=result,
                output_summary=summarize_payload(result),
            )
        return 0

    save_output = should_save_output()
    if not save_output:
        print("WORKFLOW_SAVE_OUTPUT is disabled. Results will not be written to disk.\n")

    result = test_workflow(
        message,
        output_file=args.output,
        save_output=save_output,
        run_context=run_context if should_log_runs else None,
        context_files=args.context_file,
    )

    if result and result.get("code_generator_output"):
        print("\n✅ Success! Repository JSON file generated.")
        if should_log_runs:
            run_context = result.get("_run_context", run_context)
        if should_log_runs:
            finalize_run(
                run_context,
                status="Completed",
                output_payload=result.get("code_generator_output"),
                output_summary=summarize_payload(result.get("code_generator_output", {})),
            )
        return 0
    if result:
        print("\n⚠ Workflow completed but no code generator output was produced.")
        if should_log_runs:
            run_context = result.get("_run_context", run_context)
        if should_log_runs:
            finalize_run(
                run_context,
                status="Completed",
                output_payload=result,
                output_summary="Workflow completed without code generator output.",
            )
        return 0

    print("\n❌ Workflow failed. Check the error messages above.")
    if should_log_runs:
        finalize_run(
            run_context,
            status="Failed",
            output_payload={},
            output_summary="Workflow failed.",
            error_message="Workflow failed. Check CLI logs.",
        )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except ValueError as exc:
        print(f"Input validation error: {exc}")
        raise SystemExit(1)
