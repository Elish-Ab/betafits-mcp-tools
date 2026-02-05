# Coding Style Draft

## 1. Linting

We will use **Ruff** as the unified linter and formatter. Below is sample configuration setup but can also use the extension present in the VS Code or Cursor IDE marketplace.

### Naming Conventions

- Modules, functions, and variables **must** use `snake_case`.
- Classes remain `PascalCase`.
- Constants use `UPPER_SNAKE_CASE`.
- Ruff enforces these rules; violations fail CI.

### Configuration

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
indent-width = 4
target-version = "py311"
select = ["E", "F", "I", "B", "UP", "C90"]
ignore = ["E501", "B905"]
fix = true

[tool.ruff.lint.isort]
combine-as-imports = true
force-single-line = false
```

### Rules

- Enforce consistent imports.
- Remove unused imports/variables automatically.
- Disallow wildcard imports (`from module import *`).
- Prefer f-strings over string concatenations.
- **Terminal Run**: `ruff check . --fix` (Note: this is OS dependent and can be different in MacOS/Windows/Linux).

## 2. Type Hint Extension

Enable Python's official type checking in VS Code:

- **Extension**: IDE marketplace.
- **Enable Type Checking mode**: `Ctrl+Shift+P` -> `Python: Select Type Checking Mode` -> `strict`.
- Apply hints for the public functions, classes and methods.

### Example Code

```python
def parse_pcf(file_path: str) -> dict[str, str]:
    """
    Parse PCF file and return structured data.
    
    Args:
        file_path: Path to the PCF file.
        
    Returns:
        Dictionary containing parsed PCF data.
    """
    ...
```

## 3. Testing Standards

### Testing Frameworks

- **pytest** — Standard Python testing
- **pytest-asyncio** — For asynchronous agent/chain testing
- **pytest-mock** — For mocking LLM calls and external APIs
- **coverage.py** — For code coverage analysis (goal: ≥80% core logic)

### Guide

- Mirror source structure under `/tests`
- One test file per module (e.g., `test_agents.py`)
- Use descriptive test names: `test_agent_handles_invalid_pcf`
- Mock external dependencies (OpenAI, Airtable, Slack)

### Commands

```bash
pytest -q
pytest --asyncio-mode=auto
coverage run -m pytest
coverage report -m
```

## 4. Dependencies and Environment Setup

### Dependency Management

Use **uv** (recommended) or fallback to Python's built-in venv.

#### Guide for uv:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### Guide for venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Requirements

All dependencies must be pinned in `requirements.txt` or managed via `pyproject.toml`.

## 5. Environment Configuration

Use `python-dotenv` for managing secrets and environment variables.

### Files

- `.env` (gitignored)
- `.env.example` (tracked in git)

### Usage

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

## 6. Function and Method Declaration

### Format

```python
def function_name(param1: Type, param2: Type = "default") -> ReturnType:
    """
    Short description of what the function does.
    
    Args:
        param1 (Type): Description.
        param2 (Type): Description (optional).
        
    Returns:
        ReturnType: Description of return value.
    """
    # Function logic here
```

### Guidelines

- Always include type hints and docstring
- Keep function size ≤50 lines
- Maximum 5 parameters (group with dataclass/config if more)
- Use pure functions when possible; avoid side effects

## 7. OOP or Function-Based Design

### My PCF-Workflow Directory Structure

| Layer | Style | Reason |
|-------|-------|--------|
| `agents/` | OOP | Stateful components, inheritance from BaseAgent |
| `chains/` | Hybrid / Compositional | Workflows with internal OOP |
| `graphs/` | Functional / Declarative | Stateless node functions, clean orchestration |
| `tools/` | Functional | Reusable stateless utilities |
| `models/` | OOP | Model wrappers, configuration abstraction |
| `utils/` | Functional | Helpers, pure logic functions |
| `tests/` | Functional | Clear, independent test cases |

### Examples

**OOP:**

```python
class PCFParserAgent(BaseAgent):
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
    
    def run(self, transcript: str) -> dict:
        context = self.prepare_context(transcript)
        return self.generate_output(context)
```

**Functional (Helper):**

```python
def sanitize_text(text: str) -> str:
    """Remove extra whitespace from text."""
    return re.sub(r"\s+", " ", text.strip())
```

## 8. Docstring Format

We use **Google-style docstrings**.

### Example

```python
def fetch_pcf_data(record_id: str) -> dict:
    """
    Retrieve PCF data from Airtable.
    
    Args:
        record_id (str): Airtable record ID.
        
    Returns:
        dict: PCF record details with metadata.
    """
    ...
```

### For Module Docstring (at the top of every file)

```python
"""Defines the PCF Parser Agent for LangGraph workflow integration."""
```

## 9. Coding Behaviour and Design Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| Factory Pattern | Model or agent creation | `AgentFactory.create("pcf")` |
| Singleton | Logging or Config access | Shared Logger instance |
| Decorator | Retry logic, caching | `@retry`, `@cache_result` |
| Pipeline / Chain of Responsibility | LangGraph nodes or chains | Sequential context → generation → validation |
| Adapter Pattern | Wrapping external APIs (Airtable, Slack) | `SlackAdapter`, `AirtableAdapter` |

### Behaviour Principles

- Code must be predictable, testable, and composable
- Prefer composition over inheritance
- Avoid hidden global state

## 10. File Naming Conventions

| Type | Naming | Example |
|------|--------|---------|
| Python files | `snake_case` | `base_agent.py`, `workflow_state.py` |
| Classes | `PascalCase` | `BaseAgent`, `LangGraphRunner` |
| Functions / Vars | `snake_case` | `load_env_vars()`, `context_cache` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT = 30` |
| Tests | `test_*.py` | `test_agents.py`, `test_utils.py` |

Each file should contain one logical component (one class or main function).

## 11. Error Handling Standards

### Structure

```python
try:
    result = llm.generate(context)
except TimeoutError as e:
    logger.error(f"Timeout: {e}")
    raise AgentExecutionError("LLM timed out") from e
except ValueError as e:
    logger.warning(f"Validation error: {e}")
    raise ValidationError("Invalid response") from e
```

### Custom Errors

Defined in `/utils/errors.py`:

```python
class LangGraphError(Exception):
    """Base exception for LangGraph-related errors."""
    pass

class AgentExecutionError(LangGraphError):
    """Raised when an agent execution fails."""
    pass

class ConfigError(LangGraphError):
    """Raised when configuration is invalid."""
    pass
```

### Guidelines

- Always catch known exceptions
- Add descriptive messages
- Use `raise ... from e` for stack trace preservation
- Avoid silent failures

## 12. Logging Standards

### Location

`/utils/logging.py`

### Logger Setup

```python
import logging
import json

logger = logging.getLogger("betafits")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Usage

```python
def log_event(workflow_id: str, node_id: str, event: str, details: dict = None):
    """
    Log workflow event with structured data.
    
    Args:
        workflow_id: Unique identifier for the workflow.
        node_id: Identifier for the current node.
        event: Event type or name.
        details: Optional additional event details.
    """
    payload = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "event": event,
        "details": details or {},
    }
    logger.info(json.dumps(payload))
```

### Guidelines

- Never use `print()`
- Always include context identifiers (`workflow_id`, `node_id`)
- Log levels:
  - **DEBUG** → development details
  - **INFO** → workflow progress
  - **WARNING** → recoverable issues
  - **ERROR** → critical issues



