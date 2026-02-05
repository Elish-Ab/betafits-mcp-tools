# Betafits Engineering Standard

## Purpose
This document defines the mandatory engineering standards for all Betafits backend, automation, and intelligence systems.  
It ensures consistent structure, clean separation of concerns, predictable tooling behavior, and scalable collaboration across teams and AI-assisted development.

These rules apply to **all domains**, including CRM, MCP Tools, Customer Success, Compliance, and future platforms.

---

## Core Principles

### 1. Domain-Based Repositories
- Repositories represent **business domains**, not projects or scripts.
- A domain has shared entities, workflows, and long-term ownership.
- Example domains:
  - crm
  - mcp-tools
  - customer-success
  - compliance (future)

### 2. One Repo per Domain
- A new repo is created **only** when a new business domain exists.
- Do NOT create repos for:
  - scrapers
  - LangGraph experiments
  - pipelines
  - freelancer tasks
- New repos require architectural approval.

### 3. Separation of Logic and Orchestration
- Business logic must never live in orchestration code.
- Orchestration (LangGraph, n8n) must be thin and declarative.
- Logic and orchestration must evolve independently.

### 4. Predictable, Machine-Compatible Structure
- All repos follow the same internal layout.
- Structure must be deterministic so internal AI tools (code generator, reviewer, CI) can operate reliably.

---

## Repository Naming

Repositories must follow:

apps/
services/
workflows/
lib/
tests/
docs/
pyproject.toml
README.md


Optional folders (only if needed):
infra/
scripts/
frontend/

---

## Folder Responsibilities

### apps/
**Entrypoints only.**
- API servers
- Workers
- Schedulers
- CLI runners

Rules:
- Must be thin
- Must not contain business logic
- May call workflows or services

---

### services/
**All business logic lives here.**

Includes:
- Scrapers
- Enrichment modules
- Classifiers
- Matching engines
- Decision engines
- External system adapters

Rules:
- Can import from `lib/`
- Must NOT import from `workflows/`
- Must NOT depend on `apps/`

---

### workflows/
**Orchestration only.**

Includes:
- LangGraph graphs
- n8n workflows
- Routing logic
- Fan-out / fan-in coordination

Rules:
- Must NOT contain business logic
- May call services
- Must live under:


workflows/langgraph/<flow_name>/


LangGraph flow structure:


graph.py
nodes/
<node_name>.py


---

### lib/
**Shared foundations and contracts.**

Includes:
- Canonical state models
- Pydantic models
- Enums and types
- Config loaders
- Logging and tracing
- Validation utilities
- Shared clients (DB, Airtable, LLM, RAG, KG)

Rules:
- lib/ must NOT import from services or workflows
- services and workflows may import lib

---

### tests/
Must mirror the repo structure:


tests/
services/
workflows/
lib/


Rules:
- Services require unit tests
- Workflows require graph tests with mocked services
- CI must run all tests before merge

---

### docs/
Repository-specific documentation:
- System overview
- Data flow diagrams
- LangGraph diagrams
- Developer guides
- Architecture decisions

---

## Dependency Direction (Critical)

Allowed import direction:



lib → services → workflows → apps


Forbidden:
- services importing workflows
- lib importing services
- workflows implementing business logic

---

## LangGraph-Specific Rules

1. All LangGraph code lives under:


workflows/langgraph/

2. Graphs are orchestration-only.
3. Nodes:
- Call services
- Use typed models from lib
- Do not execute heavy logic
4. Graphs must be testable with mocked services.

---

## File and Naming Rules

- snake_case for files and folders
- Descriptive names only
- Every major folder must be a Python package
- One `pyproject.toml` per repo (root only)

---

## Collaboration & Git Workflow

### Folder Ownership
- Teams own specific folders (services/<module>, workflows/<flow>, etc.)
- Cross-folder changes require coordination and review

### Branch Naming


services/<module>-<feature>
workflows/<flow>-<feature>
lib-<area>-<feature>
apps/<app>-<feature>


### PR Scope
- A PR must modify files only within its declared scope
- Shared areas require explicit justification

---

## CI Enforcement

CI must enforce:
- Required folder presence
- Naming conventions
- Forbidden imports
- Test execution
- Type checking

---

## Definition of Done

A repository is compliant when:
- It follows the standard folder structure
- Logic and orchestration are separated
- There is one shared state model
- LangGraph is thin and declarative
- CI passes with no structural violations

---

## Summary

Betafits follows a **domain-based, orchestration-driven architecture** with:

- Clean separation of concerns
- Predictable structure
- Centralized contracts
- Scalable collaboration
- First-class support for LangGraph systems

These rules are mandatory for all current and future engineering work.