---
name: daily-notebook
description: >
  Use after completing any implementation change to log important findings and
  code changes in the daily notebook. Creates or updates
  docs/daily_notebook/YYYY-MM-DD.md in the project working directory.
---

# Daily Notebook Skill

After completing an implementation, write a concise entry to the daily notebook.

## Steps

1. **Determine today's date** from the session context or memory (currentDate).

2. **Resolve the notebook path**: `<project_root>/docs/daily_notebook/YYYY-MM-DD.md`
   - Create `docs/daily_notebook/` if it doesn't exist.
   - Append to an existing file; create if new.

3. **Write the entry** using this template (append, do not overwrite):

```markdown
## HH:MM — <Short title of the change>

**Files changed:** `path/to/file.py:line`

**Problem:** One sentence describing the bug or issue.

**Root cause:** The precise technical reason it was wrong.

**Fix:** What was changed and why it's correct now.

**Impact:** What this is expected to improve.
```

4. **Keep it concise**: each entry should be 5–10 lines max. Omit anything obvious or already captured in git history.

5. **Do not log**: trivial formatting fixes, config-only changes, or anything that doesn't affect model behavior or correctness.

## When to invoke

- After fixing a bug in model, loss, or training code
- After implementing a new feature that changes training behavior
- After discovering a root cause (even if fix is pending)
