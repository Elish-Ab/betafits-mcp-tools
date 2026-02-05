# Betafits MCP Tools

AI-powered automation tools that generate and review repositories directly from Airtable-managed context. The project bundles shared LangGraph orchestration, services, and utilities so internal teams can spin up reliable workflows without re-writing core infrastructure.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Capabilities](#core-capabilities)
3. [Architecture](#architecture)
4. [Repository Layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [Setup](#setup)
7. [Running the LangGraph Workflow](#running-the-langgraph-workflow)
8. [Service Entry Points](#service-entry-points)
9. [Airtable Integration](#airtable-integration)
10. [Development Guidelines](#development-guidelines)
11. [Troubleshooting](#troubleshooting)
12. [Reference Docs](#reference-docs)

---

## Overview

Betafits MCP Tools is a domain repository that standardizes how we:

- Generate full repositories that follow internal styles and Airtable schemas.
- Review existing repositories and grade them against the Betafits rubric.
- Route incoming requests through a LangGraph brain so future nodes (refactors, PCF tools, etc.) share the same state model and tooling.

The project enforces the [Betafits Engineering Standard](docs/ENGINEERING_STANDARD.md): services hold business logic, workflows orchestrate LLM calls, and `lib/` carries shared contracts (LLM wrapper, Airtable clients, persistence helpers, etc.).

---

## Core Capabilities

- **Code Generator** (`services/code_generator/run.py`)
  - Multi-step chain extracts requirements, plans architecture, builds repo/folder/file JSON, and optionally persists it back to Airtable.
  - Enforces preferred libraries from Airtable and Betafits coding style.

- **Code Reviewer** (`services/code_reviewer/run.py`)
  - Pulls repositories, folders, and files from Airtable, matches the incoming message, and assigns rubric grades. Returns structured JSON ready for Airtable write-back.

- **MCP Brain** (`workflows/langgraph/orchestrator/nodes/brain_node.py`)
  - LLM router that selects which node should handle a user message. Currently routes to the generator or reviewer and is ready for additional nodes (PCF parser/writer, refactor, etc.).

---

## Architecture

```
message ──▶ LangGraph Orchestrator ──▶ mcp_brain ──┬─▶ code_generator_node ─▶ services/code_generator
                                                   └─▶ code_reviewer_node ─▶ services/code_reviewer
```

- **LangGraph**: `workflows/langgraph/orchestrator/graph.py` defines the directed graph. State is typed in `lib/state.py`.
- **Services**: Pure business logic. Both generator and reviewer expose a `run()` function that can be imported from CLI scripts, tests, or the graph.
- **Airtable**: `lib/airtable_client.py` fetches repositories/folders/files, field mappings, IO formats, and libraries using the credentials defined in `.env`.
- **LLM**: `lib/llm_client.py` wraps OpenRouter (default `openai/gpt-4o-mini`) with retry + throttling. All prompts live in `services/<service>/prompt.py`.
- **Persistence**: `lib/code_generator_airtable_persistence.py` converts generated JSON into Airtable payloads and batches creation of repositories/folders/files.

---

## Repository Layout

```
apps/                         # All entrypoints / CLIs
  workflow_cli/
    main.py                   # LangGraph workflow runner
docs/                         # Engineering standard, coding style, rubrics
  BETAFITS_CODE_GRADING_RUBRIC.md
  CODE_GENERATOR_AND_REVIEWER_DOCUMENTATION.md
  CODING_STYLE.md
  ENGINEERING_STANDARD.md
lib/                          # Shared clients, config, persistence helpers
services/
  code_generator/             # Multi-step generation chain + prompts + IO helpers
  code_reviewer/              # Review chain and entry point
  ...                         # (pcf_parser, pcf_writer, code_refactor scaffolds)
workflows/
  langgraph/
    orchestrator/             # Graph, nodes, and chaining logic
tests/
  README.md                   # Mirror lib/services/workflows when adding tests
requirements.txt
pyproject.toml
```

Keep code inside the correct folder (lib → services → workflows). See [ENGINEERING_STANDARD.md](docs/ENGINEERING_STANDARD.md) for the import rules we enforce in CI.

---

## Prerequisites

- Python 3.12+
- `pip`, `venv`, or [`uv`](https://github.com/astral-sh/uv) for dependency management
- Airtable base provisioned with the tables referenced below
- OpenRouter API key with access to the configured model

---

## Setup

1. **Clone**

   ```bash
   git clone <repo-url>
   cd betafits
   ```

2. **Create a virtual environment**

   ```bash
   # Preferred (uv):
   uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt

   # Standard venv:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy `.env` or create `.env.example` and set:

   ```
   OPENROUTER_API_KEY=<required>
   OPENROUTER_MODEL=openai/gpt-4o-mini           # override as needed
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_SITE_URL=https://betafits.local
   OPENROUTER_APP_NAME=Betafits MCP Tools
   AIRTABLE_API_KEY=<required>
   AIRTABLE_BASE_ID=<required>
   TRANSFORMATION_PROJECTS_TABLE=<optional>      # Airtable table name for "Link to Transformation Projects"
   DOCUMENTS_BASE_ID=<optional>                 # Airtable base ID for Documents/Resources
   DOCUMENTS_TABLE_ID=<optional>                # Airtable table ID for Documents/Resources
   PCF_BASE_ID=<optional>                        # Airtable base ID for PCF records (if different)
   LG_RUNS_BASE_ID=<optional>                    # Airtable base ID for LG Runs + Snapshots
   LG_RUNS_TABLE_ID=<optional>                   # Table ID for LG Runs
   LG_SNAPSHOTS_TABLE_ID=<optional>              # Table ID for LG State Snapshots
   LG_WORKFLOWS_TABLE_ID=<optional>              # Table ID for LG Workflows (name lookup)
   LG_NODES_TABLE_ID=<optional>                  # Table ID for LG Nodes (name lookup)
   LG_WORKFLOWS_NAME_FIELD=Name                  # Name field in LG Workflows
   LG_NODES_NAME_FIELD=Name                      # Name field in LG Nodes
   LG_WORKFLOWS_NODES_FIELD=LangGraph Nodes      # Linked field containing nodes on workflows
   LG_LOGGING_MODE=auto                          # auto | full | minimal
   LG_RUNS_MIN_FIELD_NAME=Name
   LG_RUNS_MIN_FIELD_DESCRIPTION=Description
   LG_RUNS_MIN_FIELD_STATUS=Status
   LG_RUNS_MIN_FIELD_START_DATE=Start Date
   LG_RUNS_MIN_FIELD_END_DATE=End Date
   LG_SNAPSHOTS_MIN_FIELD_TITLE=Title
   LG_SNAPSHOTS_MIN_FIELD_DETAIL=Detail
   LG_SNAPSHOTS_MIN_FIELD_CREATED_DATE=Created Date
   LG_SNAPSHOTS_MIN_FIELD_RUN_LINK=Related LG Runs Entries
   LG_RUNS_FIELD_RUN_ID=Run ID
   LG_RUNS_FIELD_WORKFLOW=Workflow
   LG_RUNS_FIELD_TRIGGERED_BY=Triggered By
   LG_RUNS_FIELD_TRIGGER_SOURCE_ID=Trigger Source ID
   LG_RUNS_FIELD_STATUS=Status
   LG_RUNS_FIELD_ENVIRONMENT=Environment
   LG_RUNS_FIELD_INPUT_SUMMARY=Input Summary
   LG_RUNS_FIELD_INPUT_PAYLOAD=Input Payload (raw JSON)
   LG_RUNS_FIELD_OUTPUT_SUMMARY=Output Summary
   LG_RUNS_FIELD_OUTPUT_PAYLOAD=Output Payload (raw JSON)
   LG_RUNS_FIELD_STEP_LOG=Step Log (compact JSON array)
   LG_RUNS_FIELD_ERROR_MESSAGE=Error Message
   LG_RUNS_FIELD_ERROR_PAYLOAD=Error Payload (raw JSON)
   LG_RUNS_FIELD_ENDED_AT=Ended At
   LG_RUNS_FIELD_PCFS=PCFs
   LG_SNAPSHOTS_FIELD_SNAPSHOT_ID=Snapshot ID
   LG_SNAPSHOTS_FIELD_RUN=Run
   LG_SNAPSHOTS_FIELD_NODE=Node
   LG_SNAPSHOTS_FIELD_INDEX=Snapshot Index
   LG_SNAPSHOTS_FIELD_TYPE=Snapshot Type
   LG_SNAPSHOTS_FIELD_STATE_JSON=State Envelope JSON
   LG_SNAPSHOTS_FIELD_SCHEMA_REF=Payload Schema Ref
   LG_SNAPSHOTS_FIELD_SCHEMA_VERSION=Payload Version
   LG_SNAPSHOTS_FIELD_STATE_HASH=State Hash
   ENVIRONMENT=Local                             # Environment label for runs
   WORKFLOW_SAVE_OUTPUT=1                        # toggle workflow output file writes
   ```

4. **Verify credentials**

   - Running `python - <<'PY'` tests `lib.config` and raises early if keys are missing:

     ```bash
     python - <<'PY'
     from lib import config
     print("Config loaded successfully:", bool(config.AIRTABLE_API_KEY))
     PY
     ```

---

## Running the LangGraph Workflow

Use the provided harness in `apps/workflow_cli` to invoke the orchestrator end-to-end:

```bash
python -m apps.workflow_cli.main "Generate a FastAPI service for employee onboarding"
```

To inject local files as additional context for the generator:

```bash
python -m apps.workflow_cli.main --context-file docs/LANGGRAPH_CONTEXT.md --context-file docs/CODING_STYLE.md "Generate a FastAPI service for employee onboarding"
```

Supported context file types: `.md`, `.txt`, `.py`, `.json`, `.yaml`, `.yml`, `.docx`, `.pdf`.
DOCX/PDF extraction requires `python-docx` and `pypdf` (included in `requirements.txt`).

To persist run logs and state snapshots into the LG Runs + LG State Snapshots tables:

```bash
python -m apps.workflow_cli.main --log-runs "Generate a FastAPI service for employee onboarding"
```

If `--workflow-name`/`--workflow-id` are not provided, the logger auto-selects a workflow by matching the run's node names against the `LangGraph Nodes` links in the LG Workflows table.

What it does:

1. Builds the initial `WorkflowState`.
2. Calls `workflows/langgraph/orchestrator/graph.workflow`.
3. Saves the repository JSON to `<repo-name>.json` when `WORKFLOW_SAVE_OUTPUT` is truthy.

Tips:

- Set `WORKFLOW_SAVE_OUTPUT=0` in `.env` to skip writing artifacts.
- Inspect `workflow_output.json` for the raw LangGraph state when debugging.
- Use `python -m apps.workflow_cli.main --help` style invocation to pass different prompts quickly.

---

## Enhancing an Existing Repository

The code generator can now add functionality to an existing repo without spinning up a brand-new structure. Provide the repo name (as stored in Airtable) and the enhancement request:

```bash
python -m apps.workflow_cli.main --repo-name ROI "Add a Slack notifier service"
```

This bypasses LangGraph and invokes the generator service directly. The LLM produces only the additional folders/files required, writes them locally, and persists them back to the Airtable `Folders`/`Files` tables via `lib/code_generator_airtable_persistence`.

You can also call the service programmatically:

```python
from services.code_generator.run import run as generate_repo

generate_repo("Add Slack alerts", repo_name="ROI")
```

If `repo_name` is omitted, the service behaves exactly as before (net-new repository generation).

If you have a PCF record that includes additional context, pass it through the context engine:

```bash
python -m apps.workflow_cli.main --repo-name ROI --pcf-record-id rec123 "Add Slack alerts"
```

```python
generate_repo("Add Slack alerts", repo_name="ROI", pcf_record_id="rec123")
```

---

## Service Entry Points

Both services can be imported directly when you need to script or test specialized behavior.

```python
from services.code_generator.run import run as generate_repo
from services.code_reviewer.run import run as review_repo

generate_result = generate_repo("Create a CRM integrations service", use_chain=True, persist=True)
review_result = review_repo("Review the onboarding automation repo")
```

- `use_chain=False` in the generator falls back to a single-shot prompt (`services/code_generator/prompt.py`) and saves files under `generated_repository/`.
- Set `persist=False` to skip pushing generator output into Airtable (useful in dev environments).
- The reviewer currently always writes JSON only; Airtable persistence happens inside the chain when `persist=True`.

---

## Airtable Integration

The workflows expect the Airtable base to expose the following tables and views:

| Table / View          | Purpose                                                                    |
|-----------------------|----------------------------------------------------------------------------|
| `Repositories`, `Folders`, `Files` | Canonical data store for generated or reviewed assets.        |
| `Field Mappings`      | Shapes JSON keys (`Current Name Field`) into Airtable columns.            |
| `IO Formats`          | Declares the schema for Code Generator Output & Code Reviewer Output.     |
| `Libraries`           | Marks preferred vs. non-preferred libraries.                              |
| `Documents/Resources` | Optional source for Engineering Standard and Coding Style content.        |

Update these tables when introducing new fields or mappings; the services call helper tools in `lib/tool_registry.py` so no extra plumbing is required. If Airtable data is missing the code logs warnings and falls back to defaults, but grading/enforcement will be weaker.

If `DOCUMENTS_BASE_ID`/`DOCUMENTS_TABLE_ID` are configured, the Context Engine will pull Engineering Standard and Coding Style from the Documents/Resources table (and include any docs linked to a supplied PCF record). When unset, local docs in `docs/` are used instead.

---

## Development Guidelines

- **Coding style**: Follow [CODING_STYLE.md](docs/CODING_STYLE.md). Run `ruff check . --fix` before every commit to enforce snake_case naming, import order, and formatting.
- **Tests**: Mirror the repo layout under `tests/`. Add unit tests for services and integration tests for workflows. Run `pytest -q`.
- **Structure**: Keep business logic inside `services/`, orchestration in `workflows/`, and shared clients/models in `lib/`. Never import "up" the stack.
- **LLM prompts**: Modify prompt templates in the relevant `services/<service>/prompt.py` file and keep instructions consistent with Airtable schema + coding style.
- **Persistence**: When adding new Airtable fields, update `lib/code_generator_airtable_persistence.py` or the Field Mapping records so persistence continues to work.

---

## Security Practices

- **Input validation**: All user messages pass through `lib.validation.validate_user_message()` before invoking services to mitigate prompt-injection attempts and oversized payloads.
- **Secret management**: Secrets live in `.env` (gitignored). Copy `.env.example`, fill in credentials locally, and never commit raw keys. `lib.config` fails fast when required keys are missing.
- **Auditing**: Run `ruff check .` and `pytest -q` during code reviews. Schedule quarterly reviews of Airtable access tokens and rotate API keys when anyone leaves the team.

---

## Troubleshooting

- **Missing API key errors**: `lib.config` raises descriptive exceptions when `OPENROUTER_API_KEY`, `AIRTABLE_API_KEY`, or `AIRTABLE_BASE_ID` are unset. Double-check your `.env`.
- **Routing to `END` immediately**: The MCP brain could not classify the message. Provide more context or ensure `OPENROUTER_MODEL` is valid.
- **Airtable schema mismatch**: Update the Field Mapping records so new columns are recognized. The persistence helper logs which columns it tried to write.
- **Rate limiting**: `lib.llm_client` backs off automatically, but you may still see waits if you exceed OpenRouter quotas. Reduce batch size or slow down calls.

---

## Reference Docs

- [CODE_GENERATOR_AND_REVIEWER_DOCUMENTATION.md](docs/CODE_GENERATOR_AND_REVIEWER_DOCUMENTATION.md) – in-depth description of the generation/review chains.
- [ENGINEERING_STANDARD.md](docs/ENGINEERING_STANDARD.md) – repository shape and import rules enforced across Betafits.
- [CODING_STYLE.md](docs/CODING_STYLE.md) – linting, formatting, and documentation requirements.
- [BETAFITS_CODE_GRADING_RUBRIC.md](docs/BETAFITS_CODE_GRADING_RUBRIC.md) – rubric consumed by the reviewer chain.
- [LANGGRAPH_CONTEXT.md](docs/LANGGRAPH_CONTEXT.md) – optional node/edge/chain context injected into generator prompts.

Use these documents whenever you extend prompts, add services, or onboard new contributors.

---

Happy building!
