---
name: test
description: Run pytest suite, analyze failures, and fix broken tests
argument-hint: "[test path or pattern]"
---

# Run Tests

## Current test results
!`cd /home/chun/work/airfrans_gnn && python -m pytest tests/ -v --tb=short 2>&1 | tail -60`

## Instructions

1. Analyze the test output above
2. If `$ARGUMENTS` is provided, run only that specific test: `pytest $ARGUMENTS -v --tb=short`
3. If tests fail:
   - Read the failing test and the source module it tests
   - Identify root cause (broken import, logic bug, missing fixture, stale reference)
   - Fix the issue — prefer fixing outdated tests over changing working source code
   - Re-run to confirm the fix
4. Report concise summary: passed/failed/skipped counts

## Known Issues

- `test_continuity_loss.py` may import legacy `loss_v3` — should use `src.navier_stokes_physics_loss`
- Edge attr has two schemas: `[dist, dir_x, dir_y]` and `[dx, dy, dist]` — tests must handle both
- Physics loss tests need `prepare_airfrans_graph_for_physics()` preprocessing first
- All tests must run on CPU without real dataset files
