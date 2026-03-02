---
name: refactor
description: Refactor code to reduce duplication and improve structure following project patterns
argument-hint: "<file path or module name>"
---

# Refactor: $ARGUMENTS

## Steps

1. Read the target file(s) completely
2. Check [known-issues.md](known-issues.md) for pre-identified problems
3. Plan changes — list files to modify BEFORE editing
4. Apply changes incrementally
5. Run `pytest tests/ -v --tb=short` to verify nothing breaks
6. If no tests exist for this module, note it and suggest `/add-test`

## Rules

### Do
- Extract duplicated code into shared modules
- Break functions over 60 lines into helpers
- Replace magic numbers with config fields or named constants
- Add type hints to modified function signatures
- Keep imports sorted: stdlib → third-party → local (`src.`)

### Do NOT
- Change public API without grepping all callers first
- Add features or change behavior — structure only
- Remove `SmokeCfg` fields without checking notebook usage
- Reformat untouched code
