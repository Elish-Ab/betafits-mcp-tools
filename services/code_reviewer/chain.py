"""LangChain pipeline for code review with proper repository/folder/file relationships.

Data Flow:
- Files have "Repositories (From Folders)" field to get related repositories
- Folders have "Repositories (Link to Repositories)" field linking to repositories
- Files have "Folders (Link To Folders)" to link to their folder
- The reviewer selects ONE best repository based on user message
- Reviews at THREE levels: repository, folder, and file
- Uses: BETAFITS_CODE_GRADING_RUBRIC.md, CODING_STYLE.md, ENGINEERING_STANDARD.md
"""

from typing import Dict, Any, List, Optional
import json
from lib.llm_client import call_llm
from lib.context_engine import build_reviewer_context, DEFAULT_PCF_TABLE
from services.code_reviewer.persistence import (
    persist_files_core_fields,
    persist_repository_core_fields,
    persist_review_results,
)


def extract_json_from_text(text: str) -> Any:
    """Extract JSON from text response."""
    import re
    json_pattern = r'```json\s*(\{.*?\}|\[.*?\])\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    try:
        start_idx = text.find('{')
        if start_idx != -1:
            depth = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return None


def prepare_context(
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
) -> Dict[str, Any]:
    """Prepare all context needed for code review."""
    return build_reviewer_context(pcf_record_id=pcf_record_id, pcf_table=pcf_table)


def build_repository_with_files(
    repo_record: Dict[str, Any],
    folders: List[Dict[str, Any]],
    files: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a complete repository structure with its related folders and files.
    
    Relationships used:
    - Folders have "Repositories (Link to Repositories)" field
    - Files have "Repositories (From Folders)" field
    - Files have "Folders (Link To Folders)" field
    """
    repo_id = repo_record.get("id")
    repo_fields = repo_record.get("fields", {})
    
    repo_name = None
    for name_field in ["Name", "Repository Name", "name", "repository_name"]:
        if name_field in repo_fields:
            repo_name = repo_fields[name_field]
            break
    
    # Find folders linked to this repository via "Repositories (Link to Repositories)"
    repo_folders = []
    folder_ids_in_repo = set()
    
    for folder in folders:
        folder_fields = folder.get("fields", {})
        folder_id = folder.get("id")
        
        linked_repos = folder_fields.get("Repositories (Link to Repositories)", [])
        if not isinstance(linked_repos, list):
            linked_repos = [linked_repos] if linked_repos else []
        
        if repo_id in linked_repos:
            folder_ids_in_repo.add(folder_id)
            repo_folders.append({
                "id": folder_id,
                "fields": folder_fields,
                "files": []  # Will be populated below
            })
    
    # Find files via "Repositories (From Folders)" field
    repo_files = []
    for file_record in files:
        file_fields = file_record.get("fields", {})
        file_id = file_record.get("id")
        
        repos_from_folders = file_fields.get("Repositories (From Folders)", [])
        if not isinstance(repos_from_folders, list):
            repos_from_folders = [repos_from_folders] if repos_from_folders else []
        
        if repo_id in repos_from_folders:
            repo_files.append({"id": file_id, "fields": file_fields})
            
            # Also add to folder's files list
            file_folder_link = file_fields.get("Folders (Link To Folders)", [])
            if not isinstance(file_folder_link, list):
                file_folder_link = [file_folder_link] if file_folder_link else []
            
            for folder in repo_folders:
                if folder["id"] in file_folder_link:
                    folder["files"].append({"id": file_id, "fields": file_fields})
    
    return {
        "id": repo_id,
        "name": repo_name,
        "fields": repo_fields,
        "folders": repo_folders,
        "files": repo_files,
    }


def step1_select_best_repository(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Step 1: Select the BEST (single) repository matching the user's message."""
    repositories = context.get("repositories", [])
    folders = context.get("folders", [])
    files = context.get("files", [])
    
    if not repositories:
        return {"error": "No repositories found in Airtable", "selected_repository": None}
    
    repo_summaries = []
    for repo in repositories:
        repo_fields = repo.get("fields", {})
        repo_id = repo.get("id")
        
        name = None
        for name_field in ["Name", "Repository Name", "name", "repository_name"]:
            if name_field in repo_fields:
                name = repo_fields[name_field]
                break
        
        description = repo_fields.get("Description", repo_fields.get("description", ""))
        
        repo_with_files = build_repository_with_files(repo, folders, files)
        folder_count = len(repo_with_files.get("folders", []))
        file_count = len(repo_with_files.get("files", []))
        
        repo_summaries.append({
            "id": repo_id,
            "name": name or "Unknown",
            "description": description[:200] if description else "",
            "folder_count": folder_count,
            "file_count": file_count,
        })

    pcf_context = context.get("pcf_context")
    pcf_context_text = json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."
    pcf_documents_summary = context.get("pcf_documents_summary", "") or "None provided."
    
    prompt = f"""You are a repository selection assistant. Select ONE repository that best matches the user's review request.

USER'S REQUEST:
{message}

PCF CONTEXT (OPTIONAL):
{pcf_context_text}

PCF DOCUMENTS (OPTIONAL):
{pcf_documents_summary}

AVAILABLE REPOSITORIES:
{json.dumps(repo_summaries, indent=2)}

INSTRUCTIONS:
1. Analyze the user's message to understand what they want to review
2. Match to the most relevant repository by name (exact or partial match)
3. Select ONLY ONE repository - the BEST match
4. If user mentions "ROI", prefer exact match "ROI" over partial matches like "ROI-Workbook-Platform"

Respond with ONLY JSON:
```json
{{
    "selected_repository_id": "<id>",
    "repository_name": "<name>",
    "selection_reason": "<reason>"
}}
```
"""
    
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    selection = extract_json_from_text(response)
    
    if not selection or "selected_repository_id" not in selection:
        # Fallback: find exact name match first
        for repo in repo_summaries:
            repo_name_lower = (repo.get("name") or "").lower()
            msg_lower = message.lower().strip()
            if repo_name_lower == msg_lower or f"review {repo_name_lower}" in msg_lower or f"review this {repo_name_lower}" in msg_lower:
                selection = {
                    "selected_repository_id": repo["id"],
                    "repository_name": repo["name"],
                    "selection_reason": "Exact name match"
                }
                break
        
        if not selection:
            best_repo = max(repo_summaries, key=lambda r: r.get("file_count", 0))
            selection = {
                "selected_repository_id": best_repo["id"],
                "repository_name": best_repo["name"],
                "selection_reason": "Fallback selection"
            }
    
    selected_repo_id = selection["selected_repository_id"]
    selected_repo = None
    for repo in repositories:
        if repo.get("id") == selected_repo_id:
            selected_repo = repo
            break
    
    if not selected_repo:
        for repo in repositories:
            repo_fields = repo.get("fields", {})
            for name_field in ["Name", "Repository Name", "name", "repository_name"]:
                if name_field in repo_fields:
                    if repo_fields[name_field] == selection.get("repository_name"):
                        selected_repo = repo
                        break
            if selected_repo:
                break
    
    if not selected_repo and repositories:
        selected_repo = repositories[0]
        selection["selection_reason"] = "Fallback to first repository"
    
    complete_repo = build_repository_with_files(selected_repo, folders, files) if selected_repo else None
    
    return {
        "selected_repository_id": selected_repo.get("id") if selected_repo else None,
        "selected_repository": complete_repo,
        "selection_info": selection,
    }


def step2_grade_repository(selected_repo: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Grade the selected repository at THREE levels:
    - Repository level (overall grades)
    - Folder level (grades and assessment per folder)
    - File level (bugginess and improvement suggestions per file)
    
    Uses:
    - BETAFITS_CODE_GRADING_RUBRIC.md for grading criteria
    - CODING_STYLE.md for style compliance
    - ENGINEERING_STANDARD.md for architecture and structure compliance
    """
    if not selected_repo:
        return {"error": "No repository selected for review"}
    
    grading_rubric = context.get("grading_rubric", "")
    coding_style = context.get("coding_style", "")
    engineering_standard = context.get("engineering_standard", "")
    preferred_libraries = context.get("preferred_libraries", [])
    pcf_context = context.get("pcf_context")
    pcf_context_text = json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."
    pcf_documents_summary = context.get("pcf_documents_summary", "") or "None provided."
    
    preferred_lib_names = []
    for lib in preferred_libraries:
        fields = lib.get("fields", {})
        for name_field in ["Name", "Library Name", "Package Name", "name"]:
            if name_field in fields:
                preferred_lib_names.append(fields[name_field])
                break
    
    repo_name = selected_repo.get("name", "Unknown Repository")
    repo_fields = selected_repo.get("fields", {})
    repo_folders = selected_repo.get("folders", [])
    repo_files = selected_repo.get("files", [])
    
    # Build folder summaries with their files
    folder_summaries = []
    for folder in repo_folders:
        folder_fields = folder.get("fields", {})
        folder_files = folder.get("files", [])
        
        file_list = []
        for f in folder_files[:5]:
            ff = f.get("fields", {})
            file_list.append({
                "file_name": ff.get("File Name", ff.get("Name", "unknown")),
                "language": ff.get("Language", ""),
            })
        
        folder_path = folder_fields.get("Folder Path", folder_fields.get("Path", ""))
        if not folder_path:
            folder_path = folder_fields.get("Folder: ROI", folder_fields.get("Folder", "Unknown"))
        
        folder_summaries.append({
            "folder_id": folder.get("id"),
            "path": folder_path,
            "description": (folder_fields.get("Description", "") or "")[:100],
            "file_count": len(folder_files),
            "sample_files": file_list,
        })
    
    # Build file summaries and capture actual source details for evaluation
    file_summaries = []
    file_loc_map: Dict[str, int] = {}
    max_files_for_prompt = 40
    max_preview_chars = 1200
    for file_record in repo_files[:max_files_for_prompt]:
        file_fields = file_record.get("fields", {})
        source_code = (
            file_fields.get("Source Code")
            or file_fields.get("source_code")
            or ""
        )
        source_code = source_code or ""
        trimmed_source = source_code.strip()
        preview = trimmed_source[:max_preview_chars]
        if trimmed_source and len(trimmed_source) > max_preview_chars:
            remaining = len(trimmed_source) - max_preview_chars
            preview = f"{preview}\n... [truncated {remaining} chars]"
        lines_of_code_field = file_fields.get("Lines of Code")
        if isinstance(lines_of_code_field, int) and lines_of_code_field > 0:
            loc_value = lines_of_code_field
        elif trimmed_source:
            loc_value = sum(1 for line in trimmed_source.splitlines() if line.strip())
        else:
            loc_value = 0
        file_id = file_record.get("id")
        file_loc_map[file_id] = loc_value
        file_summaries.append({
            "file_id": file_id,
            "file_name": file_fields.get("File Name", file_fields.get("Name", "unknown")),
            "file_path": file_fields.get("Relative File Path", file_fields.get("Path", "")),
            "language": file_fields.get("Language", ""),
            "lines_of_code": loc_value,
            "source_code_excerpt": preview,
        })
    
    prompt = f"""You are an expert code reviewer evaluating a repository against Betafits standards.

=== BETAFITS CODE GRADING RUBRIC ===
{grading_rubric}

=== BETAFITS ENGINEERING STANDARD ===
{engineering_standard[:2500]}

=== CODING STYLE GUIDELINES ===
{coding_style[:1000] if coding_style else "Standard best practices"}

=== PREFERRED LIBRARIES ===
{json.dumps(preferred_lib_names[:15], indent=2)}

=== PCF CONTEXT (OPTIONAL) ===
{pcf_context_text}

=== PCF DOCUMENTS (OPTIONAL) ===
{pcf_documents_summary}

=== REPOSITORY TO REVIEW ===
Name: {repo_name}
ID: {selected_repo.get("id")}
Description: {repo_fields.get("Description", "No description")}

FOLDERS ({len(folder_summaries)} total):
{json.dumps(folder_summaries, indent=2)}

FILES ({len(file_summaries)} of {len(repo_files)} total):
{json.dumps(file_summaries, indent=2)}

Each file entry includes a `source_code_excerpt` (first {max_preview_chars} chars of the actual code). Use that snippet to evaluate bugginess and improvement opportunities. The `lines_of_code` value has been computed directly from the source and should be reused in your output.

=== REVIEW INSTRUCTIONS ===
Review at THREE levels:

1. REPOSITORY level:
   - Overall grade (A-F) and grades for all 9 rubric areas
   - Check compliance with Engineering Standard (folder structure, separation of concerns)
   - Check if logic and orchestration are properly separated

2. FOLDER level:
   - Grade each folder (A-F)
   - Check if folder follows Engineering Standard structure (apps/, services/, workflows/, lib/, tests/)
   - Verify folder responsibilities match the standard

3. FILE level:
   - Bugginess assessment (low/medium/high)
   - Improvement suggestions
   - Check naming conventions (snake_case)

Respond with ONLY this JSON:
```json
{{
    "repo_id": "{selected_repo.get("id")}",
    "name": "{repo_name}",
    "overall_grade": "<A-F>",
    "summary": "<1-2 sentence summary including Engineering Standard compliance>",
    "suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"],
    "libraries_used": [],
    "security_flags": [],
    "engineering_standard_compliance": {{
        "folder_structure": "<compliant|partial|non-compliant>",
        "separation_of_concerns": "<compliant|partial|non-compliant>",
        "naming_conventions": "<compliant|partial|non-compliant>",
        "notes": "<brief note on compliance>"
    }},
    "grades": {{
        "repo_structure": "<grade>",
        "code_quality": "<grade>",
        "documentation": "<grade>",
        "security_practices": "<grade>",
        "modularity_testability": "<grade>",
        "dependency_management": "<grade>",
        "performance": "<grade>",
        "error_handling": "<grade>",
        "style_guide_compliance": "<grade>"
    }},
    "folders": [
        {{
            "folder_id": "<id>",
            "folder_path": "<path>",
            "overall_grade": "<grade>",
            "summary": "<1 sentence including Engineering Standard compliance>",
            "file_count": <num>
        }}
    ],
    "files": [
        {{
            "file_id": "<id>",
            "file_path": "<path>",
            "file_name": "<name>",
            "language": "<lang>",
            "bugginess": "<low|medium|high>",
            "room_for_improvement": "<suggestion>",
            "lines_of_code": <num>
        }}
    ]
}}
```
"""
    
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    review = extract_json_from_text(response)
    
    if review and isinstance(review, dict):
        for file_entry in review.get("files", []):
            file_id = file_entry.get("file_id")
            if file_id in file_loc_map:
                file_entry["lines_of_code"] = file_loc_map[file_id]
    
    if not review:
        # Fallback structure
        review = {
            "repo_id": selected_repo.get("id"),
            "name": repo_name,
            "overall_grade": "C",
            "summary": "Unable to generate detailed review.",
            "suggestions": ["Manual review recommended"],
            "libraries_used": [],
            "security_flags": [],
            "engineering_standard_compliance": {
                "folder_structure": "unknown",
                "separation_of_concerns": "unknown",
                "naming_conventions": "unknown",
                "notes": "Review required"
            },
            "grades": {
                "repo_structure": "C", "code_quality": "C", "documentation": "C",
                "security_practices": "C", "modularity_testability": "C",
                "dependency_management": "C", "performance": "C",
                "error_handling": "C", "style_guide_compliance": "C"
            },
            "folders": [
                {
                    "folder_id": f.get("id"),
                    "folder_path": f.get("fields", {}).get("Folder Path", "Unknown"),
                    "overall_grade": "C",
                    "summary": "Review required",
                    "file_count": len(f.get("files", []))
                }
                for f in repo_folders
            ],
            "files": [
                {
                    "file_id": f.get("id"),
                    "file_path": f.get("fields", {}).get("Relative File Path", ""),
                    "file_name": f.get("fields", {}).get("File Name", "unknown"),
                    "language": f.get("fields", {}).get("Language", ""),
                    "bugginess": "medium",
                    "room_for_improvement": "Review required",
                    "lines_of_code": file_loc_map.get(f.get("id"), f.get("fields", {}).get("Lines of Code", 0))
                }
                for f in repo_files[:30]
            ]
        }
    
    return review


def run_code_reviewer_chain(
    message: str,
    persist_to_airtable: bool = True,
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
) -> Dict[str, Any]:
    """
    Run the complete code reviewer chain.
    
    Reviews at THREE levels using:
    - BETAFITS_CODE_GRADING_RUBRIC.md
    - CODING_STYLE.md
    - ENGINEERING_STANDARD.md
    
    Levels:
    1. Repository level - overall grades + Engineering Standard compliance
    2. Folder level - grades and assessment per folder
    3. File level - bugginess and improvement suggestions
    """
    print("Code Reviewer: Preparing context...")
    context = prepare_context(pcf_record_id=pcf_record_id, pcf_table=pcf_table)
    
    print(f"  - Repositories: {len(context.get('repositories', []))}")
    print(f"  - Folders: {len(context.get('folders', []))}")
    print(f"  - Files: {len(context.get('files', []))}")
    print(f"  - Grading Rubric: {'Loaded' if context.get('grading_rubric') else 'Not found'}")
    print(f"  - Coding Style: {'Loaded' if context.get('coding_style') else 'Not found'}")
    print(f"  - Engineering Standard: {'Loaded' if context.get('engineering_standard') else 'Not found'}")
    print(f"  - PCF Context: {'Loaded' if context.get('pcf_context') else 'Not found'}")
    print(f"  - PCF Documents: {len(context.get('pcf_documents', []))}")
    
    print("\nStep 1: Selecting best repository...")
    selection_result = step1_select_best_repository(message, context)
    
    if selection_result.get("error"):
        return {
            "error": selection_result["error"],
            "selected_repository_ids": [],
            "review": None,
            "context_used": {}
        }
    
    selected_repo = selection_result.get("selected_repository")
    selection_info = selection_result.get("selection_info", {})
    
    print(f"  Selected: {selected_repo.get('name') if selected_repo else 'None'}")
    print(f"  Reason: {selection_info.get('selection_reason', 'N/A')}")
    print(f"  Folders in repo: {len(selected_repo.get('folders', [])) if selected_repo else 0}")
    print(f"  Files in repo: {len(selected_repo.get('files', [])) if selected_repo else 0}")
    
    print("\nStep 2: Grading repository (repo/folder/file levels)...")
    review = step2_grade_repository(selected_repo, context)

    if review.get("error"):
        return {
            "error": review["error"],
            "selected_repository_ids": [selection_result.get("selected_repository_id")],
            "review": None,
            "context_used": {}
        }

    print(f"  Overall Grade: {review.get('overall_grade', 'N/A')}")
    print(f"  Engineering Standard: {review.get('engineering_standard_compliance', {}).get('folder_structure', 'N/A')}")
    print(f"  Folders reviewed: {len(review.get('folders', []))}")
    print(f"  Files reviewed: {len(review.get('files', []))}")

    airtable_updates: Dict[str, int] = {}
    if persist_to_airtable:
        try:
            if persist_repository_core_fields(review):
                airtable_updates["repositories"] = airtable_updates.get("repositories", 0) + 1
        except Exception as exc:
            print(f"  Warning: Failed to persist repository core fields: {exc}")

        try:
            file_count = persist_files_core_fields(review)
            if file_count:
                airtable_updates["files"] = airtable_updates.get("files", 0) + file_count
        except Exception as exc:
            print(f"  Warning: Failed to persist file core fields: {exc}")

        try:
            mapping_updates = persist_review_results(review, context.get("field_mappings", []))
            if mapping_updates:
                for key, value in mapping_updates.items():
                    airtable_updates[key] = airtable_updates.get(key, 0) + value
        except Exception as exc:
            print(f"  Warning: Failed to persist review via field mappings: {exc}")

        if airtable_updates:
            print(f"  Airtable updates: {airtable_updates}")

    return {
        "selected_repository_ids": [selection_result.get("selected_repository_id")],
        "review": {
            "repositories": [review]
        },
        "context_used": {
            "repositories_count": len(context.get("repositories", [])),
            "total_folders_count": len(context.get("folders", [])),
            "total_files_count": len(context.get("files", [])),
            "standards_used": ["BETAFITS_CODE_GRADING_RUBRIC.md", "CODING_STYLE.md", "ENGINEERING_STANDARD.md"],
            "per_repository_filtered": {
                review.get("name", "Unknown"): {
                    "folders_count": len(selected_repo.get("folders", [])) if selected_repo else 0,
                    "files_count": len(selected_repo.get("files", [])) if selected_repo else 0,
                }
            },
            "airtable_updates": airtable_updates,
            "pcf_context_present": bool(context.get("pcf_context")),
            "pcf_context_error": context.get("pcf_context_error"),
            "pcf_documents_count": len(context.get("pcf_documents", [])),
            "documents_context_error": context.get("documents_context_error"),
        }
    }
