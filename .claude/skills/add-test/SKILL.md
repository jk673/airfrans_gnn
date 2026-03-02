---
name: add-test
description: Write pytest tests for a specified source module
argument-hint: "<module name, e.g. navier_stokes_physics_loss>"
---

# Add Tests for: $ARGUMENTS

## Steps

1. Read the target module `src/$ARGUMENTS.py` thoroughly
2. Read existing tests and [test-patterns.md](test-patterns.md) for conventions
3. Write tests in `tests/test_$0.py`
4. Run `pytest tests/test_$0.py -v --tb=short` to verify all pass

## Rules

- Create **minimal synthetic PyG `Data` objects** — never load real dataset files
- Use `torch.allclose(result, expected, atol=1e-5)` for float comparisons
- Test names: `test_<function>_<scenario>`
- All tests must run on CPU
- Add shared fixtures to `tests/conftest.py`
