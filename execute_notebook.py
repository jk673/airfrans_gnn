#!/usr/bin/env python3
"""Execute a Jupyter notebook programmatically."""

import json
import sys
import traceback
from pathlib import Path


def execute_notebook(notebook_path):
    """Execute all cells in a Jupyter notebook."""
    print(f"Loading notebook: {notebook_path}")

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    print(f"Found {len(nb['cells'])} cells")

    # Create a namespace for execution
    namespace = {}

    code_cells = [i for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code']
    print(f"Executing {len(code_cells)} code cells...\n")

    for idx, cell_num in enumerate(code_cells, 1):
        cell = nb['cells'][cell_num]
        source = ''.join(cell['source'])

        # Skip empty cells
        if not source.strip():
            print(f"[{idx}/{len(code_cells)}] Cell {cell_num+1}: Empty, skipping")
            continue

        # Show first line of cell
        first_line = source.split('\n')[0][:80]
        print(f"[{idx}/{len(code_cells)}] Cell {cell_num+1}: {first_line}...")

        try:
            exec(source, namespace)
            print(f"  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            print(f"\n--- Cell source ---")
            print(source[:500])
            print(f"--- End source ---\n")
            traceback.print_exc()
            return False

        print()

    print("✓ All cells executed successfully!")
    return True


if __name__ == '__main__':
    notebook_path = '01_trainer_executed.ipynb'
    success = execute_notebook(notebook_path)
    sys.exit(0 if success else 1)
