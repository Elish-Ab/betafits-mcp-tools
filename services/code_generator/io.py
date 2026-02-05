"""Input/Output handling for code generator service."""
from typing import Dict, Any, List
import json
from pathlib import Path


def parse_output(output: str) -> Dict[str, Any]:
    """
    Parse the LLM output into structured format with code files.
    
    Args:
        output: Raw output from LLM
        
    Returns:
        Dictionary with folder structure, code files, and JSON files
    """
    import re
    
    result = {
        "folder_structure": [],
        "code_files": {},
        "json_files": {},
        "raw_output": output,
    }
    
    # Extract code blocks (Python, JavaScript, TypeScript, etc.)
    code_pattern = r'```(\w+)?\n(.*?)```'
    code_matches = re.findall(code_pattern, output, re.DOTALL)
    
    file_counter = {}
    for lang, code_content in code_matches:
        lang = lang.lower() if lang else "txt"
        code_content = code_content.strip()
        
        # Determine file extension
        ext_map = {
            "python": "py",
            "py": "py",
            "javascript": "js",
            "js": "js",
            "typescript": "ts",
            "ts": "ts",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "markdown": "md",
            "md": "md",
            "txt": "txt",
            "html": "html",
            "css": "css",
            "shell": "sh",
            "bash": "sh",
        }
        ext = ext_map.get(lang, "txt")
        
        # Try to extract filename from context or use default
        filename = None
        lines_before = output[:output.find(f"```{lang}")].split('\n')[-5:]
        for line in reversed(lines_before):
            if '/' in line or '\\' in line:
                # Try to extract filename from path
                potential_name = line.strip().split('/')[-1].split('\\')[-1]
                if '.' in potential_name:
                    filename = potential_name
                    break
        
        if not filename:
            # Generate filename based on language and counter
            if lang == "json":
                counter = file_counter.get("json", 0)
                file_counter["json"] = counter + 1
                filename = f"data_{counter}.json" if counter > 0 else "data.json"
            else:
                counter = file_counter.get(lang, 0)
                file_counter[lang] = counter + 1
                filename = f"file_{counter}.{ext}" if counter > 0 else f"main.{ext}"
        
        # Store code file
        if ext == "json":
            try:
                parsed_json = json.loads(code_content)
                result["json_files"][filename] = parsed_json
            except json.JSONDecodeError:
                # If JSON parsing fails, store as text
                result["code_files"][filename] = code_content
        else:
            result["code_files"][filename] = code_content
    
    # Extract folder structure from markdown-style tree
    lines = output.split('\n')
    for line in lines:
        stripped = line.strip()
        # Look for folder structure indicators
        if (stripped.startswith('├──') or 
            stripped.startswith('└──') or 
            stripped.startswith('│') or
            (stripped.startswith('/') and '/' in stripped) or
            (stripped.startswith('./') and '/' in stripped)):
            # Clean up the line
            clean_line = stripped.replace('├──', '').replace('└──', '').replace('│', '').strip()
            if clean_line and (clean_line.startswith('/') or clean_line.startswith('./') or '/' in clean_line):
                result["folder_structure"].append(clean_line)
    
    return result


def format_output_for_airtable(parsed_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the parsed output for Airtable insertion.
    
    Args:
        parsed_output: Parsed output dictionary
        
    Returns:
        Formatted dictionary ready for Airtable
    """
    return {
        "folder_structure": json.dumps(parsed_output.get("folder_structure", [])),
        "code_files": json.dumps(list(parsed_output.get("code_files", {}).keys())),
        "json_files": json.dumps(parsed_output.get("json_files", {})),
        "raw_output": parsed_output.get("raw_output", ""),
    }


def save_output_to_files(output: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    Save the generated output to files, creating the full repository structure.
    
    Args:
        output: Parsed output dictionary
        output_dir: Directory to save files to
        
    Returns:
        List of created file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    # Save code files
    code_files = output.get("code_files", {})
    for filename, content in code_files.items():
        # Handle nested paths
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(file_path)
    
    # Save JSON files
    json_files = output.get("json_files", {})
    for filename, content in json_files.items():
        # Handle nested paths
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        created_files.append(file_path)
    
    # Save folder structure
    structure_file = output_dir / "folder_structure.txt"
    with open(structure_file, 'w', encoding='utf-8') as f:
        for line in output.get("folder_structure", []):
            f.write(line + '\n')
    created_files.append(structure_file)
    
    # Save raw output
    raw_file = output_dir / "raw_output.txt"
    with open(raw_file, 'w', encoding='utf-8') as f:
        f.write(output.get("raw_output", ""))
    created_files.append(raw_file)
    
    return created_files

