#!/usr/bin/env python3
"""
Execute a notebook's code cells headlessly (no nbconvert needed).

Figures and tables are written by the cells themselves; cell outputs are not
stored back into the notebook. Open the notebook in Jupyter/VS Code to see
them interactively.

Usage:
    MPLBACKEND=Agg python analysis/run_notebook.py analysis/paper_analysis.ipynb
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "analysis/paper_analysis.ipynb").resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))

    os.chdir(path.parent)
    namespace = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        started = time.time()

        try:
            exec(compile(source, f"{path.name}[cell {index}]", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            print(f"\nFAILED at cell {index}")
            sys.exit(1)

        print(f"[cell {index:02d}] ok ({time.time() - started:.1f}s)")

    print("NOTEBOOK COMPLETE")


if __name__ == "__main__":
    main()
