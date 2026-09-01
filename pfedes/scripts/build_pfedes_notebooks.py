from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    builder = Path(__file__).resolve().parents[2] / "scripts" / "build_common_notebooks.py"
    subprocess.run([sys.executable, str(builder), "--method", "pfedes"], check=True)
