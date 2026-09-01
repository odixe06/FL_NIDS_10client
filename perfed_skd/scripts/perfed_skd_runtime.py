from __future__ import annotations

import importlib.util
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[2] / "scripts" / "common_client_parallel_runtime.py"
SPEC = importlib.util.spec_from_file_location("task10_common_runtime", RUNTIME)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

if __name__ == "__main__":
    MODULE.main()
