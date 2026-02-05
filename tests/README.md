# Tests

Mirror the production code layout under `tests/` whenever you add coverage:

```
tests/
  lib/
  services/
  workflows/
```

- **Unit tests** for shared helpers and services live under the matching path (e.g., `tests/services/code_generator/`).
- **Graph tests** should exercise LangGraph nodes with mocked services to keep orchestration thin.
- Run `pytest -q` from the repository root; CI expects this directory structure.
