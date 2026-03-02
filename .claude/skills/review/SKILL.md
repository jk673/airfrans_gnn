---
name: review
description: Code review following project conventions for physics-informed GNN code
argument-hint: "[file path or leave empty for uncommitted changes]"
context: fork
agent: general-purpose
---

# Code Review

## Target
If `$ARGUMENTS` is a file path, review that file.
If empty, review all uncommitted changes below.

## Uncommitted changes
!`cd /home/chun/work/airfrans_gnn && git diff --stat && echo "---DIFF---" && git diff && git diff --cached`

## Checklist

### Correctness
- Tensor shapes consistent (especially batch dimension with PyG `batch` tensor)
- Edge attr format handled: both `[dist, dir_x, dir_y]` and `[dx, dy, dist]`
- Normalization/denormalization in correct order and direction
- `scatter` operations specify `dim_size`
- `edge_index` is `[2, E]` LongTensor

### Numerical Stability
- Division guarded (area floors, epsilon in denominators)
- No unbounded values to `log()`, `sqrt()`, `exp()`
- NaN/Inf checks in loss computation

### Code Quality
- No duplicated functions across modules (`_extract_dxdy_length`, `_half_edges`, `_valid_edges` are known duplicates)
- No bare `except Exception:` — use specific types
- No hardcoded magic numbers — use config or named constants
- No in-place ops on gradient-requiring tensors

### Project Conventions
- Source in `src/`, preprocessing in `preprocessing/`, tests in `tests/`
- Config via dataclass fields, not function arg defaults
- Comments in English (flag mixed Korean/English)

## Output Format

```
## Summary
[1-2 sentences]

## Issues
### [CRITICAL/HIGH/MEDIUM/LOW]: [title]
- File: path:line
- Problem: ...
- Fix: ...

## Looks Good
- [positive observations]
```
