"""Prompt templates for code generation."""
from typing import Dict, Any, List
import json


def build_step1_extract_requirements_prompt(message: str, context: Dict[str, Any]) -> str:
    """
    Build prompt for Step 1: Extract requirements from user message.
    
    Args:
        message: User's input message
        context: Context with Airtable fields and coding style
        
    Returns:
        Formatted prompt string
    """
    repositories_fields = context.get("repositories_fields", {})
    folders_fields = context.get("folders_fields", {})
    files_fields = context.get("files_fields", {})
    coding_style = context.get("coding_style", "")
    engineering_standard = context.get("engineering_standard", "")
    langgraph_context = context.get("langgraph_context", "")
    pcf_context = context.get("pcf_context")
    pcf_documents_summary = context.get("pcf_documents_summary", "")
    extra_context_files_summary = context.get("extra_context_files_summary", "")
    
    prompt = f"""Convert the following user intention into a structured Software Requirements Specification (SRS).

## Available Context:

### Airtable Table Fields:

**Repositories Table Fields:**
{repositories_fields}

**Folders Table Fields:**
{folders_fields}

**Files Table Fields:**
{files_fields}

### Coding Style Guidelines:
{coding_style}

### Engineering Standard (MANDATORY STRUCTURE REQUIREMENTS):
{engineering_standard}

### LangGraph Context (optional):
{langgraph_context or "None provided."}

### PCF Context (optional):
{json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."}

### PCF Documents (optional):
{pcf_documents_summary or "None provided."}

### Additional File Context (optional):
{extra_context_files_summary or "None provided."}

## User Intention:
{message}

## Task:
Provide a JSON SRS with the following structure:
{{
  "functional_requirements": ["requirement1", "requirement2", ...],
  "non_functional_requirements": ["requirement1", "requirement2", ...],
  "constraints": ["constraint1", "constraint2", ...],
  "domain_models": [
    {{"name": "Entity1", "description": "...", "fields": ["field1", "field2"]}},
    ...
  ]
}}

Use the Airtable fields to understand what data structures and fields should be included.
All repository plans produced later MUST comply with the Engineering Standard above (directory layout, separation of orchestration vs. services vs. lib, naming conventions, testing expectations).
Return ONLY valid JSON, no additional text."""
    
    return prompt


def build_step2_plan_architecture_prompt(srs: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Build prompt for Step 2: Plan system architecture based on SRS.
    
    Args:
        srs: Software Requirements Specification
        context: Context with Airtable fields and libraries
        
    Returns:
        Formatted prompt string
    """
    repositories_fields = context.get("repositories_fields", {})
    folders_fields = context.get("folders_fields", {})
    files_fields = context.get("files_fields", {})
    io_format = context.get("io_format", {})
    preferred_libraries = context.get("preferred_libraries", [])
    non_preferred_libraries = context.get("non_preferred_libraries", [])
    langgraph_context = context.get("langgraph_context", "")
    pcf_context = context.get("pcf_context")
    pcf_documents_summary = context.get("pcf_documents_summary", "")
    extra_context_files_summary = context.get("extra_context_files_summary", "")
    
    srs_text = json.dumps(srs, indent=2)
    
    # Extract library names from preferred libraries
    preferred_lib_names = []
    for lib in preferred_libraries:
        fields = lib.get("fields", {})
        # Try common field names for library name
        lib_name = None
        for name_field in ["Name", "Library Name", "Package Name", "name", "library_name", "package_name"]:
            if name_field in fields:
                lib_name = fields[name_field]
                break
        if lib_name:
            preferred_lib_names.append(lib_name)
    
    # Extract library names from non-preferred (for reference/warning)
    non_preferred_lib_names = []
    for lib in non_preferred_libraries:
        fields = lib.get("fields", {})
        for name_field in ["Name", "Library Name", "Package Name", "name", "library_name", "package_name"]:
            if name_field in fields:
                non_preferred_lib_names.append(fields[name_field])
                break
    
    preferred_libs_text = json.dumps(preferred_libraries, indent=2, default=str)
    
    prompt = f"""Design a system architecture based on this SRS:

{srs_text}

## Available Context:

### Airtable Table Fields:

**Repositories Table Fields:**
{repositories_fields}

**Folders Table Fields:**
{folders_fields}

**Files Table Fields:**
{files_fields}

### CRITICAL: Preferred Libraries (MUST USE ONLY THESE)

**Preferred Libraries:**
{preferred_libs_text}

**Preferred Library Names:**
{', '.join(preferred_lib_names) if preferred_lib_names else 'None found'}

**Non-Preferred Libraries (DO NOT USE):**
{', '.join(non_preferred_lib_names) if non_preferred_lib_names else 'None'}

### Output Format Specification:
{io_format}

### LangGraph Context (optional):
{langgraph_context or "None provided."}

### PCF Context (optional):
{json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."}

### PCF Documents (optional):
{pcf_documents_summary or "None provided."}

### Additional File Context (optional):
{extra_context_files_summary or "None provided."}

## Task:
Provide a JSON architecture plan with:
{{
  "modules": ["module1", "module2", ...],
  "services": ["service1", "service2", ...],
  "data_flows": ["flow1", "flow2", ...],
  "libraries": ["library1", "library2", ...],
  "dependencies": [
    {{"from": "module1", "to": "module2", "type": "import"}},
    ...
  ]
}}

CRITICAL REQUIREMENTS:
1. **ONLY use libraries from the Preferred Libraries list above**
2. **DO NOT use any library that is NOT in the Preferred Libraries list**
3. If you need a library that is not in the preferred list, you MUST find an alternative from the preferred list
4. The "libraries" array should ONLY contain library names from the Preferred Libraries list
5. Consider the Airtable fields when designing data structures
6. All external dependencies must be from the Preferred Libraries list
7. Architecture must align with the Engineering Standard (apps/, services/, workflows/, lib/, infra/, tests/, orchestration boundaries, etc.)

Return ONLY valid JSON, no additional text."""
    
    return prompt


def build_step3_create_repository_structure_prompt(
    architecture: Dict[str, Any],
    context: Dict[str, Any],
    message: str
) -> str:
    """
    Build prompt for Step 3: Create hierarchical repository structure.
    
    Args:
        architecture: Architecture plan
        context: Context with Airtable fields and Field Mappings
        message: User's original message
        
    Returns:
        Formatted prompt string
    """
    field_mappings = context.get("field_mappings", [])
    io_format = context.get("io_format", {})
    coding_style = context.get("coding_style", "")
    engineering_standard = context.get("engineering_standard", "")
    langgraph_context = context.get("langgraph_context", "")
    pcf_context = context.get("pcf_context")
    pcf_documents_summary = context.get("pcf_documents_summary", "")
    extra_context_files_summary = context.get("extra_context_files_summary", "")
    repositories_fields = context.get("repositories_fields", {})
    folders_fields = context.get("folders_fields", {})
    files_fields = context.get("files_fields", {})
    
    architecture_text = json.dumps(architecture, indent=2)
    
    # Organize Field Mappings by entity type
    repo_fields = []
    folder_fields = []
    file_fields = []
    
    for mapping in field_mappings:
        fields = mapping.get("fields", {})
        json_pattern = fields.get("JSON Pattern", "").lower()
        current_name = fields.get("Current Name Field", "")
        
        # Determine which entity this field belongs to
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
    
    # Sort each entity's fields by Field Order
    repo_fields.sort(key=lambda x: x.get("Field Order", 999))
    folder_fields.sort(key=lambda x: x.get("Field Order", 999))
    file_fields.sort(key=lambda x: x.get("Field Order", 999))
    
    field_mappings_text = json.dumps({
        "repositories_fields": repo_fields,
        "folders_fields": folder_fields,
        "files_fields": file_fields,
    }, indent=2)
    
    prompt = f"""Generate a hierarchical repository structure as JSON for a project based on this architecture:

{architecture_text}

## CRITICAL: Field Mappings Specification

You MUST create a hierarchical JSON structure with three main arrays: repositories, folders, and files.
Only include fields specified in Field Mappings where Action Type = "Populate".

{field_mappings_text}

### Field Mappings Rules:
1. **ONLY** output fields where Action Type = "Populate" (already filtered for you above)
2. Use **Current Name Field** as the JSON key (exact name, case-sensitive)
3. Follow **JSON Pattern** exactly (this defines the structure/nesting)
4. Fill the value based on **LLM Notes** (instructions for what value to generate)
5. Output fields in **ascending Field Order** (already sorted for you)
6. **DO NOT create any other fields** - only use the fields specified in Field Mappings

### Available Airtable Table Fields (for reference):
**Repositories Table Fields:**
{json.dumps(repositories_fields, indent=2)}

**Folders Table Fields:**
{json.dumps(folders_fields, indent=2)}

**Files Table Fields:**
{json.dumps(files_fields, indent=2)}

### Output Format Specification:
{io_format}

### Coding Style Guidelines:
{coding_style}

### Engineering Standard (REPOSITORY STRUCTURE RULES):
{engineering_standard}

### LangGraph Context (optional):
{langgraph_context or "None provided."}

### PCF Context (optional):
{json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."}

### PCF Documents (optional):
{pcf_documents_summary or "None provided."}

### Additional File Context (optional):
{extra_context_files_summary or "None provided."}

### User Request:
{message}

## Task:
Generate a hierarchical repository structure JSON with this EXACT structure:

{{
  "repositories": [
    {{
      // Use ONLY fields from repositories_fields in Field Mappings above
      // Use Current Name Field as key, follow JSON Pattern, fill from LLM Notes
      // **IMPORTANT: Generate a proper repository name from the user's message for the "Name" field**
      // The repository name should be descriptive, kebab-case, and based on the project description
      // Example: "fastapi-web-api", "python-data-processor", "react-dashboard-app"
      // Output in Field Order
    }}
  ],
  "folders": [
    {{
      // Use ONLY fields from folders_fields in Field Mappings above
      // Use Current Name Field as key, follow JSON Pattern, fill from LLM Notes
      // Output in Field Order
    }}
  ],
  "files": [
    {{
      // Use ONLY fields from files_fields in Field Mappings above
      // Use Current Name Field as key, follow JSON Pattern, fill from LLM Notes
      // IMPORTANT: "Source Code" field must be "" (empty string) - code will be generated later
      // Output in Field Order
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. Create a hierarchical structure with repositories, folders, and files as separate arrays
2. For each entity, ONLY include fields from the corresponding Field Mappings list
3. Use Current Name Field as the JSON key (exact name, case-sensitive)
4. Follow JSON Pattern to understand nesting/structure
5. Generate values based on LLM Notes
6. **Generate a proper repository name from the user's message** - make it descriptive, kebab-case, and project-specific
7. Maintain Field Order within each entity
8. For files: "Source Code" field must be "" (empty string) initially
9. Do NOT add any fields not specified in Field Mappings
10. Repository/folder/file hierarchy must comply with the Engineering Standard (apps/, services/, workflows/, lib/, infra/, tests/, shared utilities)

Return ONLY valid JSON, no additional text."""
    
    return prompt


def build_step4_update_file_code_batch_prompt(
    files_to_generate: List[Dict[str, Any]],
    previous_files_text: str,
    available_imports: Dict[str, str],
    context: Dict[str, Any],
    strict_mode: bool = False,
    quality_notes: str = "",
) -> str:
    """
    Build prompt for Step 4: Generate code for multiple files in a batch and write the generated code to files.
    
    Args:
        files_to_generate: List of file definitions to generate code for
        previous_files_text: Text description of previously generated files
        available_imports: Dictionary of available imports
        context: Context with Airtable fields
        
    Returns:
        Formatted prompt string
    """
    repositories_fields = context.get("repositories_fields", {})
    folders_fields = context.get("folders_fields", {})
    files_fields = context.get("files_fields", {})
    io_format = context.get("io_format", {})
    coding_style = context.get("coding_style", "")
    engineering_standard = context.get("engineering_standard", "")
    langgraph_context = context.get("langgraph_context", "")
    pcf_context = context.get("pcf_context")
    pcf_documents_summary = context.get("pcf_documents_summary", "")
    extra_context_files_summary = context.get("extra_context_files_summary", "")
    
    # Build file definitions for batch
    file_definitions_text = "\n".join([
        f"""
File {idx + 1}:
- File Name: {f.get('File Name', '')}
- Relative File Path: {f.get('Relative File Path', '')}
- Language: {f.get('Language', 'Python')}
- Description: {f.get('Description', '')}
- All Airtable Fields: {json.dumps({k: v for k, v in f.items() if k != 'Source Code'}, indent=2)}"""
        for idx, f in enumerate(files_to_generate)
    ])
    
    strict_instructions = ""
    if strict_mode:
        strict_instructions = """
STRICT REGENERATION MODE:
- Replace any placeholders, stubs, or TODOs with full working implementations.
- DO NOT use: TODO, FIXME, TBD, pass, NotImplementedError, placeholder, or "raise NotImplementedError".
- Provide concrete logic and real function bodies (no pseudocode).
- Avoid demo-only or mock data unless explicitly requested.
"""

    quality_notes_text = f"\nQuality issues to fix:\n{quality_notes}\n" if quality_notes else ""

    prompt = f"""Generate complete, production-ready code for {len(files_to_generate)} files in a batch:

## Files to Generate:
{file_definitions_text}

## Available Context:

### Airtable Table Fields:

**Repositories Table Fields:**
{repositories_fields}

**Folders Table Fields:**
{folders_fields}

**Files Table Fields:**
{files_fields}

### Output Format Specification:
{io_format}

### Coding Style Guidelines:
{coding_style}

### Engineering Standard (Repository/Folders/Files MUST follow this):
{engineering_standard}

### LangGraph Context (optional):
{langgraph_context or "None provided."}

### PCF Context (optional):
{json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."}

### PCF Documents (optional):
{pcf_documents_summary or "None provided."}

### Additional File Context (optional):
{extra_context_files_summary or "None provided."}

### Previous Files (for imports):
{previous_files_text}

### Available Imports:
{json.dumps(available_imports, indent=2) if available_imports else "{}"}

{strict_instructions}{quality_notes_text}

## Instructions:
1. Generate complete, working code for ALL {len(files_to_generate)} files - not placeholders
2. Include all necessary imports (use available imports when possible)
3. Include proper type hints, docstrings, and error handling
4. Follow the coding style guidelines strictly and enforce the Engineering Standard layout (apps/, services/, workflows/, lib/, infra/, tests/)
5. Use the Airtable fields specified in "fields_to_include" to structure data
6. Ensure all expected exports are present
7. Make the code functional and production-ready
8. Files can import from each other within this batch

## Output Format:
Return a JSON object mapping file paths to their code:
{{
  "path/to/file1.py": "complete code here...",
  "path/to/file2.py": "complete code here...",
  ...
}}

Each key should be the exact file path from the file definitions above.
Return ONLY valid JSON, no markdown formatting or code blocks."""
    
    # Write the generated code to the respective files
    for file_def in files_to_generate:
        file_path = file_def.get("Relative File Path", "")
        code = file_def.get("Source Code", "")
        
        if file_path and code:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the code to the file
            with open(file_path, 'w') as file:
                file.write(code)
    
    return prompt



def build_code_generation_prompt(context: Dict[str, Any]) -> str:
    """
    Build the prompt for code generation.
    
    Args:
        context: Dictionary containing all context information
        
    Returns:
        Formatted prompt string
    """
    repositories_fields = context.get("repositories_fields", {})
    folders_fields = context.get("folders_fields", {})
    files_fields = context.get("files_fields", {})
    io_format = context.get("io_format", {})
    coding_style = context.get("coding_style", "")
    engineering_standard = context.get("engineering_standard", "")
    langgraph_context = context.get("langgraph_context", "")
    pcf_context = context.get("pcf_context")
    pcf_documents_summary = context.get("pcf_documents_summary", "")
    extra_context_files_summary = context.get("extra_context_files_summary", "")
    message = context.get("message", "")
    
    prompt = f"""You are an expert code generator that creates complete code repositories with actual code files based on repository context.

Your task is to generate a complete code repository with:
1. Full folder structure
2. Actual code files (Python, JavaScript, etc.) with real, working code
3. JSON files that match the specified output format based on Airtable fields
4. Configuration files (requirements.txt, package.json, etc.)
5. Documentation files (README.md, etc.)

## Available Context:

### Airtable Table Fields:

**Repositories Table Fields:**
{repositories_fields}

**Folders Table Fields:**
{folders_fields}

**Files Table Fields:**
{files_fields}

### Output Format Specification:
{io_format}

### Coding Style Guidelines:
{coding_style}

### Engineering Standard (Structure + Directory expectations):
{engineering_standard}

### LangGraph Context (optional):
{langgraph_context or "None provided."}

### PCF Context (optional):
{json.dumps(pcf_context, indent=2, default=str) if pcf_context else "None provided."}

### PCF Documents (optional):
{pcf_documents_summary or "None provided."}

### Additional File Context (optional):
{extra_context_files_summary or "None provided."}

## Instructions:

1. Analyze the user's message: {message}
2. Use the table fields to understand the data structure and what fields need to be included
3. Generate actual, working code files - not just placeholders
4. Create JSON files that include data based on the Airtable fields provided
5. Follow the output format specification exactly for JSON structure
6. Follow all coding style guidelines for all code files
7. Enforce the Engineering Standard structure (apps/, services/, workflows/, lib/, tests/, infra/, docs) for every folder/file you create
8. Include proper imports, error handling, and documentation
9. Generate a complete, functional code repository

## Output Format:

Provide the output in the following structure:

```
project_name/
├── file1.py (with actual code content)
├── file2.js (with actual code content)
├── config.json (with data based on Airtable fields)
├── requirements.txt
└── README.md
```

For each file, use code blocks:
- ```python
  # actual Python code here
  ```
- ```javascript
  // actual JavaScript code here
  ```
- ```json
  {{"field1": "value1", "field2": "value2"}}
  ```

IMPORTANT:
- Generate REAL, WORKING code - not pseudocode or placeholders
- Include all necessary imports and dependencies
- Follow the coding style guidelines strictly
- Adhere to the Engineering Standard structure and naming conventions
- Use the Airtable fields to populate JSON files with appropriate data
- Make the code functional and complete

Be mindful of token usage. Prioritize core functionality and essential files."""
    
    return prompt
