"""LangChain pipeline for code generation with multi-step process."""
from typing import Dict, Any, List
from pathlib import Path
import json
from lib.llm_client import call_llm
from services.code_generator.prompt import (
    build_step1_extract_requirements_prompt,
    build_step2_plan_architecture_prompt,
    build_step3_create_repository_structure_prompt,
    build_step4_update_file_code_batch_prompt,
)


def extract_json_from_text(text: str) -> Any:
    """
    Extract JSON from text response.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON object or None
    """
    import re
    # Try to find JSON in code blocks
    json_pattern = r'```json\s*(\{.*?\}|\[.*?\])\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON without code blocks
    json_pattern = r'(\{.*?\}|\[.*?\])'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def _extract_source_code(file_def: Dict[str, Any]) -> str:
    if "Source Code" in file_def:
        return file_def.get("Source Code") or ""
    for key in file_def.keys():
        if "source" in key.lower() and "code" in key.lower():
            return file_def.get(key) or ""
    return ""


def _should_enforce_quality(file_def: Dict[str, Any]) -> bool:
    file_path = file_def.get("Relative File Path", file_def.get("File Name", "")) or ""
    file_name = file_def.get("File Name", "") or file_path
    lower_name = file_name.lower()
    if lower_name in {"__init__.py", "__all__.py"}:
        return False
    ext = Path(lower_name).suffix
    if ext in {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini"}:
        return False
    return True


def _detect_quality_issues(code: str) -> List[str]:
    issues = []
    if not code or len(code.strip()) < 40:
        issues.append("too_short")
    placeholder_markers = ["TODO", "FIXME", "TBD", "placeholder", "NotImplementedError"]
    if any(marker in code for marker in placeholder_markers):
        issues.append("placeholder_markers")
    for line in code.splitlines():
        if line.strip() == "pass":
            issues.append("pass_statement")
            break
    return issues


def _collect_quality_issues(
    repository_structure: Dict[str, Any],
    file_indices: List[int],
) -> Dict[str, List[str]]:
    issues_by_file: Dict[str, List[str]] = {}
    files = repository_structure.get("files", [])
    for file_idx in file_indices:
        if file_idx >= len(files):
            continue
        file_def = files[file_idx]
        if not isinstance(file_def, dict) or not _should_enforce_quality(file_def):
            continue
        file_path = file_def.get("Relative File Path", file_def.get("File Name", "unknown"))
        code = _extract_source_code(file_def)
        if not code:
            issues_by_file[file_path] = ["missing_code"]
            continue
        issues = _detect_quality_issues(code)
        if issues:
            issues_by_file[file_path] = issues
    return issues_by_file


def step1_extract_requirements(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: Extract requirements from user message.
    
    Args:
        message: User's input message
        context: Context with Airtable fields and coding style
        
    Returns:
        Structured Software Requirements Specification (SRS)
    """
    prompt = build_step1_extract_requirements_prompt(message, context)
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    srs = extract_json_from_text(response)
    
    if not srs:
        # Fallback structure
        srs = {
            "functional_requirements": [message],
            "non_functional_requirements": [],
            "constraints": [],
            "domain_models": []
        }
    
    return srs


def step2_plan_architecture(srs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Plan system architecture based on SRS.
    Uses preferred libraries from Airtable and enforces that only preferred libraries are used.
    
    Args:
        srs: Software Requirements Specification
        context: Context with Airtable fields and libraries
        
    Returns:
        Architecture plan with libraries
    """
    # Extract library names for validation
    preferred_libraries = context.get("preferred_libraries", [])
    preferred_lib_names = []
    for lib in preferred_libraries:
        fields = lib.get("fields", {})
        for name_field in ["Name", "Library Name", "Package Name", "name", "library_name", "package_name"]:
            if name_field in fields:
                preferred_lib_names.append(fields[name_field])
                break
    
    prompt = build_step2_plan_architecture_prompt(srs, context)
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    architecture = extract_json_from_text(response)
    
    if not architecture:
        architecture = {
            "modules": ["main"],
            "services": [],
            "data_flows": [],
            "libraries": preferred_lib_names[:5] if preferred_lib_names else [],  # Use preferred libraries
            "dependencies": []
        }
    else:
        # Validate and filter libraries to ensure only preferred ones are used
        if "libraries" in architecture:
            architecture_libs = architecture["libraries"]
            # Filter to only include preferred libraries
            valid_libraries = [lib for lib in architecture_libs if lib in preferred_lib_names]
            if len(valid_libraries) != len(architecture_libs):
                print(f"⚠️  Warning: Some libraries were not in preferred list. Filtered: {architecture_libs} -> {valid_libraries}")
            architecture["libraries"] = valid_libraries
        else:
            # Add preferred libraries if not present
            architecture["libraries"] = preferred_lib_names[:5] if preferred_lib_names else []
    
    return architecture


def step3_create_repository_structure(architecture: Dict[str, Any], context: Dict[str, Any], message: str) -> Dict[str, Any]:
    """
    Step 3: Create hierarchical repository structure as JSON (repositories, folders, files).
    Uses Field Mappings to determine which fields to include in each entity.
    
    Args:
        architecture: Architecture plan
        context: Context with Airtable fields and Field Mappings
        message: User's original message (for repository name)
        
    Returns:
        Hierarchical repository structure: {{"repositories": [...], "folders": [...], "files": [...]}}
        Each entity follows Field Mappings specification
    """
    # Organize Field Mappings by entity type for fallback structure
    field_mappings = context.get("field_mappings", [])
    repo_fields = []
    folder_fields = []
    file_fields = []
    
    for mapping in field_mappings:
        fields = mapping.get("fields", {})
        json_pattern = fields.get("JSON Pattern", "").lower()
        current_name = fields.get("Current Name Field", "")
        
        if "repositories" in json_pattern or "repository" in json_pattern:
            repo_fields.append({
                "Current Name Field": current_name,
                "JSON Pattern": fields.get("JSON Pattern", ""),
                "LLM Notes": fields.get("LLM Notes", ""),
                "Field Order": fields.get("Field Order", 999),
            })
        elif "folders" in json_pattern or "folder" in json_pattern:
            folder_fields.append({
                "Current Name Field": current_name,
                "JSON Pattern": fields.get("JSON Pattern", ""),
                "LLM Notes": fields.get("LLM Notes", ""),
                "Field Order": fields.get("Field Order", 999),
            })
        elif "files" in json_pattern or "file" in json_pattern:
            file_fields.append({
                "Current Name Field": current_name,
                "JSON Pattern": fields.get("JSON Pattern", ""),
                "LLM Notes": fields.get("LLM Notes", ""),
                "Field Order": fields.get("Field Order", 999),
            })
        else:
            # Default: try to match by field name
            if any(keyword in current_name.lower() for keyword in ["repo", "repository"]):
                repo_fields.append({
                    "Current Name Field": current_name,
                    "JSON Pattern": fields.get("JSON Pattern", ""),
                    "LLM Notes": fields.get("LLM Notes", ""),
                    "Field Order": fields.get("Field Order", 999),
                })
            elif any(keyword in current_name.lower() for keyword in ["folder"]):
                folder_fields.append({
                    "Current Name Field": current_name,
                    "JSON Pattern": fields.get("JSON Pattern", ""),
                    "LLM Notes": fields.get("LLM Notes", ""),
                    "Field Order": fields.get("Field Order", 999),
                })
            elif any(keyword in current_name.lower() for keyword in ["file", "source code"]):
                file_fields.append({
                    "Current Name Field": current_name,
                    "JSON Pattern": fields.get("JSON Pattern", ""),
                    "LLM Notes": fields.get("LLM Notes", ""),
                    "Field Order": fields.get("Field Order", 999),
                })
    
    prompt = build_step3_create_repository_structure_prompt(architecture, context, message)
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    repository_structure = extract_json_from_text(response)
    
    if not repository_structure:
        # Fallback: Build hierarchical structure from Field Mappings
        repository_structure = {
            "repositories": [],
            "folders": [],
            "files": []
        }
        
        # Build a default repository entry
        repo_entry = {}
        for field in repo_fields:
            repo_entry[field["Current Name Field"]] = ""
        if repo_entry:
            repository_structure["repositories"].append(repo_entry)
        
        # Build a default folder entry
        folder_entry = {}
        for field in folder_fields:
            folder_entry[field["Current Name Field"]] = ""
        if folder_entry:
            repository_structure["folders"].append(folder_entry)
        
        # Build a default file entry
        file_entry = {}
        for field in file_fields:
            key = field["Current Name Field"]
            file_entry[key] = "" if key != "Source Code" else ""
        if file_entry:
            repository_structure["files"].append(file_entry)
    
    # Ensure structure has the three main arrays
    if not isinstance(repository_structure, dict):
        repository_structure = {"repositories": [], "folders": [], "files": []}
    
    if "repositories" not in repository_structure:
        repository_structure["repositories"] = []
    if "folders" not in repository_structure:
        repository_structure["folders"] = []
    if "files" not in repository_structure:
        repository_structure["files"] = []
    
    # Ensure "Source Code" field is empty for all files
    for file_def in repository_structure.get("files", []):
        if isinstance(file_def, dict):
            if "Source Code" in file_def:
                file_def["Source Code"] = ""
            elif "Source Code" not in file_def:
                # Check if any field name contains "Source Code" or similar
                for key in file_def.keys():
                    if "source" in key.lower() and "code" in key.lower():
                        file_def[key] = ""
    
    return repository_structure


def step4_update_file_code_batch(
    repository_structure: Dict[str, Any],
    file_indices: List[int],
    context: Dict[str, Any],
    strict_mode: bool = False,
    quality_notes: str = "",
) -> Dict[str, Any]:
    """
    Step 4: Update "Source Code" fields for multiple files in the repository structure (batch processing).
    
    Args:
        repository_structure: Complete repository structure matching Airtable schema (will be updated in place)
        file_indices: List of file indices to update in repository_structure["files"]
        context: Context with Airtable fields
        
    Returns:
        Updated repository structure with "Source Code" filled in for the specified files
    """
    files = repository_structure.get("files", [])
    
    # Get files to generate
    files_to_generate = [files[i] for i in file_indices if i < len(files)]
    
    # Build context about files with code already generated (using Airtable field names)
    previous_files = []
    available_imports = {}
    for i, file_def in enumerate(files):
        # Find the correct field name for source code (could be "Source Code", "files.code", etc.)
        source_code = None
        if "Source Code" in file_def:
            source_code = file_def.get("Source Code", "")
        else:
            # Look for any field containing "source" and "code"
            for key in file_def.keys():
                if "source" in key.lower() and "code" in key.lower():
                    source_code = file_def.get(key, "")
                    break
        
        if source_code and i not in file_indices:  # Files with code, not in current batch
            file_path = file_def.get("Relative File Path", file_def.get("File Name", ""))
            file_name = file_def.get("File Name", "")
            previous_files.append(f"- {file_path}: {file_name}")
            # Extract potential exports from file name or path
            if file_name:
                module_name = file_name.replace(".py", "").replace(".", "_")
                available_imports[module_name] = file_path.replace("/", ".").replace(".py", "")
    
    previous_files_text = "\n".join(previous_files) if previous_files else "None"
    
    prompt = build_step4_update_file_code_batch_prompt(
        files_to_generate=files_to_generate,
        previous_files_text=previous_files_text,
        available_imports=available_imports,
        context=context,
        strict_mode=strict_mode,
        quality_notes=quality_notes,
    )
    response = call_llm(prompt, model="google/gemini-2.0-flash-001")
    code_map = extract_json_from_text(response)
    
    # Update repository structure with generated code
    # Find the correct field name for source code (could be "Source Code", "files.code", etc.)
    if code_map and isinstance(code_map, dict):
        for i, file_idx in enumerate(file_indices):
            if file_idx < len(files):
                file_def = files[file_idx]
                file_path = file_def.get("Relative File Path", file_def.get("File Name", ""))
                
                # Try to find code for this file
                code = None
                if file_path in code_map:
                    code = code_map[file_path]
                else:
                    # Try to match by file name
                    file_name = file_def.get("File Name", "")
                    if file_name in code_map:
                        code = code_map[file_name]
                    else:
                        # Try to match by index if path doesn't match
                        code_keys = list(code_map.keys())
                        if i < len(code_keys):
                            code = code_map[code_keys[i]]
                
                if code:
                    # Clean up code (remove markdown if present)
                    if "```" in code:
                        import re
                        code_pattern = r'```(?:\w+)?\n(.*?)```'
                        match = re.search(code_pattern, code, re.DOTALL)
                        if match:
                            code = match.group(1).strip()
                    
                    # Find the correct field name for source code
                    # Check for "Source Code" first (common case)
                    if "Source Code" in file_def:
                        code_field_name = "Source Code"
                    else:
                        # Look for any field containing "source" and "code" (handles "files.code", etc.)
                        code_field_name = None
                        for key in file_def.keys():
                            if "source" in key.lower() and "code" in key.lower():
                                code_field_name = key
                                break
                        # If still not found, default to "Source Code" and create it
                        if not code_field_name:
                            code_field_name = "Source Code"
                    
                    # Update the code field
                    repository_structure["files"][file_idx][code_field_name] = code
    
    return repository_structure


def run_code_generator_chain(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete multi-step code generator chain.
    
    Args:
        context: Context dictionary with all required fields
        
    Returns:
        Complete code generation result with SRS, architecture, structure, and files
    """
    message = context.get("message", "")
    
    # Step 1: Extract requirements
    print("Step 1: Extracting requirements...")
    srs = step1_extract_requirements(message, context)
    
    # Step 2: Plan architecture
    print("Step 2: Planning architecture...")
    architecture = step2_plan_architecture(srs, context)
    
    # Step 3: Create complete repository structure (with empty "Source Code" fields)
    print("Step 3: Creating repository structure...")
    repository_structure = step3_create_repository_structure(architecture, context, message)
    
    # Step 4: Update "Source Code" field for ALL files in batches
    # IMPORTANT: Always process files in batches to minimize LLM API calls
    # Note: Structure is now based on Field Mappings, so we need to find files differently
    files = []
    if isinstance(repository_structure, dict):
        # Try to find files array (could be nested based on Field Mappings)
        if "files" in repository_structure and isinstance(repository_structure["files"], list):
            files = repository_structure["files"]
        # Also check if repository_structure itself contains file-like structures
        # This depends on how Field Mappings structure the output
    
    BATCH_SIZE = 3  # Generate 3 files per LLM call (adjust based on rate limits)
    
    if not files:
        print("Step 4: No files to generate code for (structure may not contain files array).")
    else:
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
        
        print(f"Step 4: Generating code for {len(files)} files in {total_batches} batches (batch size: {BATCH_SIZE})...")
        print(f"  ⚠️  Processing in batches to minimize LLM calls - never processing files individually.")
        
        # Process ALL files in batches - each batch makes ONE LLM call for multiple files
        processed_count = 0
        failed_batches = []
        
        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(files))
            batch_indices = list(range(batch_start, batch_end))
            batch_files = [files[i].get("Relative File Path", files[i].get("File Name", "")) for i in batch_indices]
            
            print(f"  Batch {batch_num + 1}/{total_batches} ({len(batch_indices)} files): {', '.join(batch_files)}")
            
            try:
                # This function makes ONE LLM call to generate code for ALL files in batch_indices
                repository_structure = step4_update_file_code_batch(
                    repository_structure=repository_structure,
                    file_indices=batch_indices,  # Multiple files processed in single LLM call
                    context=context
                )
                quality_issues = _collect_quality_issues(repository_structure, batch_indices)
                if quality_issues:
                    quality_notes = "\n".join(
                        f"{path}: {', '.join(issues)}"
                        for path, issues in quality_issues.items()
                    )
                    print("  ⚠️  Detected low-quality code, regenerating batch with strict mode...")
                    repository_structure = step4_update_file_code_batch(
                        repository_structure=repository_structure,
                        file_indices=batch_indices,
                        context=context,
                        strict_mode=True,
                        quality_notes=quality_notes,
                    )
                processed_count += len(batch_indices)
            except Exception as e:
                print(f"  ⚠️  Error generating batch {batch_num + 1}: {e}")
                print(f"     Skipping {len(batch_indices)} files in this batch, continuing with next batch...")
                failed_batches.append(batch_num + 1)
                # Continue with other batches - don't stop processing
                continue
        
        print(f"  ✓ Completed: {processed_count}/{len(files)} files processed")
        if failed_batches:
            print(f"  ⚠️  Failed batches: {failed_batches} ({len(files) - processed_count} files not generated)")
    
    # Build final result
    # Structure is hierarchical: repositories, folders, files
    repositories = repository_structure.get("repositories", [])
    folders = repository_structure.get("folders", [])
    files_list = repository_structure.get("files", [])
    
    # Count files with code (check for any field containing "source" and "code")
    def has_source_code(file_def):
        if not isinstance(file_def, dict):
            return False
        # Check for "Source Code" field first
        if file_def.get("Source Code"):
            return True
        # Look for any field containing "source" and "code"
        for key in file_def.keys():
            if "source" in key.lower() and "code" in key.lower() and file_def.get(key):
                return True
        return False
    
    files_with_code = [f for f in files_list if has_source_code(f)]
    
    # Extract repository name from first repository entry
    repo_name = "unknown"
    if repositories and isinstance(repositories[0], dict):
        # Try common field names for repository name
        for key in ["Name", "Repository Name", "name", "repository_name"]:
            if key in repositories[0]:
                repo_name = repositories[0][key]
                break
    
    result = {
        "srs": srs,
        "architecture": architecture,
        "repository": repository_structure,  # Hierarchical structure: {repositories: [], folders: [], files: []}
        "summary": {
            "repository_name": repo_name,
            "structure_type": "hierarchical",
            "total_repositories": len(repositories),
            "total_folders": len(folders),
            "total_files": len(files_list),
            "files_with_code": len(files_with_code),
            "field_mappings_count": len(context.get("field_mappings", []))
        }
    }
    
    return result
