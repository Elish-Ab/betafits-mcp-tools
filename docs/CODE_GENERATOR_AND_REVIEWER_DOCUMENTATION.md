# Code Generator and Code Reviewer Implementation Documentation

## Overview

This document provides a comprehensive overview of the implementation of two core services in the Betafits MCP Tools system: the **Code Generator** and the **Code Reviewer**. Both services are integrated into a LangGraph-based workflow orchestrator and interact with Airtable for context and data management.

---

## Code Generator Service

### Purpose
The Code Generator service automatically generates complete software repositories from natural language descriptions. It creates structured codebases following Betafits coding standards, using preferred libraries, and organizing code according to Airtable field mappings.

### Architecture

The Code Generator uses a **multi-step chain architecture** that breaks down code generation into four distinct phases:

#### Step 1: Extract Requirements
- **Function**: `step1_extract_requirements()`
- **Purpose**: Analyzes the user's natural language message and extracts structured requirements
- **Output**: Software Requirements Specification (SRS) containing:
  - Functional requirements
  - Non-functional requirements
  - Constraints
  - Domain models

#### Step 2: Plan Architecture
- **Function**: `step2_plan_architecture()`
- **Purpose**: Designs the system architecture based on the SRS
- **Key Features**:
  - **Library Validation**: Enforces that only preferred libraries from Airtable are used
  - **Automatic Filtering**: Removes any non-preferred libraries suggested by the LLM
  - **Fallback**: Uses preferred libraries if architecture planning fails
- **Output**: Architecture plan with:
  - Modules and services
  - Data flows
  - Libraries (validated to be preferred only)
  - Dependencies

#### Step 3: Create Repository Structure
- **Function**: `step3_create_repository_structure()`
- **Purpose**: Creates a hierarchical repository structure matching Airtable schema
- **Key Features**:
  - Uses Field Mappings from Airtable to determine which fields to include
  - Creates three main entities: repositories, folders, and files
  - Follows Field Order from mappings for proper field sequencing
  - Initializes "Source Code" fields as empty (to be populated in Step 4)
- **Output**: Complete repository structure in JSON format:
  ```json
  {
    "repositories": [...],
    "folders": [...],
    "files": [...]
  }
  ```

#### Step 4: Generate Source Code (Batch Processing)
- **Function**: `step4_update_file_code_batch()`
- **Purpose**: Generates actual source code for all files in the repository
- **Key Features**:
  - **Batch Processing**: Processes multiple files per LLM call (default: 3 files per batch)
  - **Efficiency**: Minimizes API calls by never processing files individually
  - **Context Awareness**: Uses previously generated files for imports and dependencies
  - **Error Resilience**: Continues processing other batches if one fails
- **Output**: Complete repository structure with all "Source Code" fields populated

### Context Preparation

The service prepares comprehensive context from multiple sources:

1. **Airtable Tables**:
   - Repositories fields schema
   - Folders fields schema
   - Files fields schema

2. **IO Format**:
   - Code Generator Output Format from Airtable
   - Defines the expected JSON structure

3. **Field Mappings**:
   - Filtered by Code Generator IO Format
   - Only includes mappings with Action Type = "Populate"
   - Sorted by Field Order (ascending)

4. **Libraries**:
   - All libraries from Airtable
   - Separated into preferred and non-preferred
   - Preferred libraries are enforced during architecture planning

5. **Coding Style**:
   - Loads `CODING_STYLE.md` from the project root
   - Ensures generated code follows Betafits standards

### Integration Points

- **Entry Point**: `services/code_generator/run.py::run()`
- **Chain Implementation**: `workflows/langgraph/orchestrator/chains/code_generator_chain.py`
- **Workflow Node**: `workflows/langgraph/orchestrator/nodes/code_generator.py`
- **IO Handling**: `services/code_generator/io.py` (for fallback mode)

### Output Format

The service returns a complete JSON structure containing:

```json
{
  "srs": { /* Software Requirements Specification */ },
  "architecture": { /* System architecture plan */ },
  "repository": {
    "repositories": [ /* Repository records */ ],
    "folders": [ /* Folder records */ ],
    "files": [ /* File records with source code */ ]
  },
  "summary": {
    "repository_name": "...",
    "total_repositories": 1,
    "total_folders": N,
    "total_files": M,
    "files_with_code": M
  },
  "context_used": { /* Metadata about context sources */ }
}
```

### Key Implementation Details

1. **Library Enforcement**: The system automatically validates and filters libraries to ensure only preferred ones are used, preventing the use of non-preferred libraries.

2. **Batch Processing**: Code generation is optimized to process multiple files per LLM call, reducing API costs and improving efficiency.

3. **Field Mapping Compliance**: All generated structures strictly follow Airtable Field Mappings, ensuring compatibility with the database schema.

4. **Error Handling**: The system gracefully handles failures in individual batches, continuing with remaining files.

---

## Code Reviewer Service

### Purpose
The Code Reviewer service analyzes existing code repositories stored in Airtable and grades them according to the Betafits Code Grading Rubric. It provides detailed feedback on code quality, security, documentation, and adherence to coding standards.

### Architecture

The Code Reviewer uses a **two-step chain architecture**:

#### Step 1: Select Repositories
- **Function**: `step1_select_repositories()`
- **Purpose**: Intelligently matches the user's message to relevant repositories in Airtable
- **Process**:
  - Fetches all repositories from Airtable
  - Uses LLM to analyze user message and repository metadata
  - Returns list of repository IDs that should be reviewed
- **Output**: List of selected repository IDs

#### Step 2: Grade Repositories
- **Function**: `step2_grade_repositories()`
- **Purpose**: Performs comprehensive code review and grading
- **Key Features**:
  - **Per-File Analysis**: Analyzes each file in the repository
  - **Bugginess Assessment**: Assigns risk level (low/medium/high) per file
  - **Improvement Suggestions**: Provides specific recommendations per file
  - **Rubric-Based Grading**: Grades repositories across 9 areas:
    1. Repo Structure
    2. Code Quality
    3. Documentation
    4. Security Practices
    5. Modularity/Testability
    6. Dependency Management
    7. Performance
    8. Error Handling
    9. Style Guide Compliance
  - **Library Analysis**: Identifies libraries used and flags non-preferred ones
  - **Security Flags**: Highlights potential security issues
- **Output**: Complete review JSON with grades and recommendations

### Context Preparation

The service gathers context from:

1. **Airtable Data**:
   - All repositories
   - All folders (linked to repositories)
   - All files (linked to repositories via folders or directly)

2. **Grading Rubric**:
   - Loads `BETAFITS_CODE_GRADING_RUBRIC.md`
   - Uses as authoritative standard for grading

3. **Coding Style**:
   - Loads `CODING_STYLE.md`
   - Used for style-related grading

4. **IO Format**:
   - Code Reviewer Output Format from Airtable
   - Defines the expected review output structure

5. **Field Mappings**:
   - Filtered for Code Reviewer Output
   - Only Action Type = "Populate"
   - Sorted by Field Order

6. **Libraries**:
   - Preferred vs non-preferred libraries
   - Used to flag non-preferred library usage

### Grading System

The reviewer assigns letter grades (A, B, C, D, F) with optional +/- modifiers:

- **A (Excellent)**: Best practices throughout, production-ready
- **B (Good)**: Mostly solid, minor improvements needed
- **C (Average)**: Functional but with notable deficiencies
- **D (Poor)**: Significant problems, not maintainable
- **F (Fail)**: Critically flawed, needs rewrite

### Per-File Analysis

For each file, the reviewer provides:

1. **Bugginess Level**: Risk assessment (low/medium/high)
2. **Room for Improvement**: 1-2 sentence recommendations

### Repository-Level Analysis

For each repository, the reviewer provides:

1. **Overall Grade**: Summary grade for the repository
2. **Area Grades**: Individual grades for each of the 9 rubric areas
3. **Libraries Used**: List of detected libraries/packages
4. **Security Flags**: List of potential security issues
5. **Summary**: 1-2 sentence overview
6. **Suggestions**: 1-3 improvement recommendations

### Integration Points

- **Entry Point**: `services/code_reviewer/run.py::run()`
- **Chain Implementation**: `workflows/langgraph/orchestrator/chains/code_reviewer_chain.py`
- **Workflow Node**: `workflows/langgraph/orchestrator/nodes/code_reviewer.py`

### Output Format

The service returns:

```json
{
  "selected_repository_ids": ["repo_id_1", ...],
  "review": {
    "repositories": [
      {
        /* Repository review record following Field Mappings */
        "overall_grade": "B+",
        "repo_structure": "A",
        "code_quality": "B",
        "documentation": "C",
        "security_practices": "B",
        "modularity_testability": "B",
        "dependency_management": "A",
        "performance": "B",
        "error_handling": "B",
        "style_guide_compliance": "A",
        "libraries_used": ["library1", "library2"],
        "security_flags": [],
        "summary": "...",
        "suggestions": ["...", "..."],
        "files": [
          {
            "bugginess": "low",
            "room_for_improvement": "...",
            /* Other file fields per Field Mappings */
          }
        ]
      }
    ]
  },
  "context_used": {
    "repositories_count": N,
    "folders_count": M,
    "files_count": K,
    "field_mappings_count": L,
    "libraries_count": P
  }
}
```

### Key Implementation Details

1. **Metadata-Based Review**: The reviewer works from Airtable metadata (file paths, languages, descriptions) rather than full source code, making it efficient for large repositories.

2. **Intelligent Repository Selection**: Uses LLM to match user queries to relevant repositories, even with partial or natural language descriptions.

3. **Relationship Mapping**: Correctly maps files to repositories through folder relationships or direct links, handling various Airtable schema configurations.

4. **Conservative Grading**: When information is uncertain, grades conservatively (towards C) to avoid false positives.

5. **Library Preference Enforcement**: Flags non-preferred libraries in security flags or suggestions, ensuring alignment with Betafits standards.

6. **Field Mapping Compliance**: Review output strictly follows Airtable Field Mappings, ensuring compatibility with database schema.

---

## Common Features

Both services share several important characteristics:

### 1. Airtable Integration
- Both services fetch context from Airtable tables
- Use Field Mappings to ensure output matches database schema
- Respect IO Format specifications for structured output

### 2. LLM-Based Processing
- Both use Gemini 2.5 Flash Lite model for LLM calls
- Implement JSON extraction from LLM responses
- Include fallback mechanisms for parsing failures

### 3. Library Management
- Both services work with preferred/non-preferred library lists
- Code Generator enforces preferred libraries
- Code Reviewer flags non-preferred library usage

### 4. Coding Standards
- Both reference `CODING_STYLE.md` for style guidelines
- Code Generator generates code following these standards
- Code Reviewer grades adherence to these standards

### 5. Error Handling
- Both services include comprehensive error handling
- Graceful degradation when context is missing
- Continue processing even if individual steps fail

### 6. Workflow Integration
- Both are integrated as nodes in the LangGraph orchestrator
- Can be invoked through the MCP Brain routing system
- Return structured state updates for workflow continuation

---

## Technical Stack

- **Language**: Python 3.12+
- **LLM**: Google Gemini 2.5 Flash Lite
- **Orchestration**: LangGraph
- **Database**: Airtable (via pyairtable)
- **Architecture**: Multi-step chain pattern

---

## File Structure

```
services/
├── code_generator/
│   ├── run.py              # Main entry point
│   ├── prompt.py           # Prompt templates
│   └── io.py               # Input/output handling
│
└── code_reviewer/
    ├── run.py              # Main entry point
    ├── prompt.py           # Prompt templates (empty, using chain)
    └── io.py               # Input/output handling (empty, using chain)

workflows/langgraph/orchestrator/
├── chains/
│   ├── code_generator_chain.py    # Multi-step code generation chain
│   └── code_reviewer_chain.py     # Multi-step code review chain
│
└── nodes/
    ├── code_generator.py          # Workflow node for code generation
    └── code_reviewer.py           # Workflow node for code review
```

---

## Usage Examples

### Code Generator

```python
from services.code_generator.run import run

result = run(
    message="Create a Python REST API for managing user accounts with authentication",
    use_chain=True
)

# Result contains:
# - srs: Software Requirements Specification
# - architecture: System architecture plan
# - repository: Complete repository structure with code
# - summary: Statistics about generated repository
```

### Code Reviewer

```python
from services.code_reviewer.run import run

result = run(
    message="Review the authentication service repository"
)

# Result contains:
# - selected_repository_ids: IDs of repositories to review
# - review: Complete review with grades and recommendations
# - context_used: Metadata about context sources
```

---

## Future Enhancements

### Code Generator
- Support for incremental code generation (updating existing repositories)
- Integration with version control systems
- Support for multiple programming languages beyond Python
- Template-based generation for common patterns

### Code Reviewer
- Direct Airtable write-back of review results
- Historical tracking of code quality trends
- Integration with CI/CD pipelines
- Support for reviewing actual source code (not just metadata)

---

## Conclusion

Both the Code Generator and Code Reviewer services represent sophisticated implementations that leverage LLM capabilities, Airtable integration, and structured workflows to automate software development tasks. They follow Betafits coding standards, enforce library preferences, and provide comprehensive, structured outputs that integrate seamlessly with the Airtable database schema.

The multi-step chain architecture ensures that complex tasks are broken down into manageable steps, with proper context preparation, error handling, and output validation at each stage.


