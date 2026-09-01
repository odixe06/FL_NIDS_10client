#!/usr/bin/env python3
"""Validate the structural DDP contract of generated Kaggle notebooks."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


REQUIRED_TOKENS = {
    "DDP import": "DistributedDataParallel",
    "DDP wrapper": "DDP(",
    "distributed package": "torch.distributed",
    "NCCL backend": "nccl",
    "torchrun launcher": "torch.distributed.run",
    "loopback rendezvous": "--master-addr=127.0.0.1",
    "dynamic master port": "--master-port=",
    "two-process launch": "--nproc-per-node=",
    "rank environment": "RANK",
    "local-rank environment": "LOCAL_RANK",
    "world-size environment": "WORLD_SIZE",
    "rank-local device": "torch.cuda.set_device(local_rank)",
    "process group": "init_process_group",
    "device-bound process group": "device_id=device",
    "device-bound barrier": "barrier(device_ids=[local_rank])",
    "legacy NCCL variable removal": "launch_environment.pop(legacy_async_key",
    "current NCCL async-error variable": "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL loopback interface": '"NCCL_SOCKET_IFNAME": "lo"',
    "training sampler": "DistributedSampler",
    "sampler epoch": "set_epoch(",
    "mixed precision": "autocast(",
    "gradient scaler": "GradScaler(",
    "best checkpoint": "best.pt",
    "last checkpoint": "last.pt",
    "rank-zero log": "run.log",
    "nonzero-rank log": "rank_1.log",
    "traceback logging": "traceback.format_exc()",
    "Kaggle input": "/kaggle/input",
    "Kaggle output": "/kaggle/working",
    "Kaggle temporary storage": "/kaggle/temp",
}
FORBIDDEN_PATTERNS = {
    "DataParallel wrapper": re.compile(r"(?<!Distributed)DataParallel\s*\("),
    "invalid logging grouping": re.compile(r"%(?:\([^)]+\))?[-+ #0]*,"),
    "unresolved TODO": re.compile(r"\bTODO\b"),
    "unresolved confirmed placeholder": re.compile(r"\bCONFIRMED[_-]"),
    "deprecated NCCL async-error variable": re.compile(
        r"(?<!TORCH_)NCCL_ASYNC_ERROR_HANDLING"
    ),
    "standalone hostname rendezvous": re.compile(r"--standalone"),
    "device-implicit barrier": re.compile(r"\bbarrier\(\s*\)"),
}


def code_cells(notebook: dict):
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            yield index, "".join(cell.get("source", []))


def embedded_runtime_sources(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name)
            and (
                target.id.endswith("RUNTIME_SOURCE")
                or target.id.endswith("SCRIPT_SOURCE")
            )
            for target in targets
        ):
            yield value.value


def validate_notebook(path: Path) -> list[str]:
    failures = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"invalid notebook JSON: {error}"]

    if notebook.get("nbformat") != 4:
        failures.append("nbformat must be 4")
    cells = list(code_cells(notebook))
    combined_source = "\n".join(source for _, source in cells)
    embedded_runtimes = []

    for cell_index, source in cells:
        if source.lstrip().startswith(("!", "%")):
            continue
        try:
            tree = ast.parse(source, filename=f"{path}:cell_{cell_index}")
        except SyntaxError as error:
            failures.append(
                f"cell {cell_index} is not valid ordinary Python: {error}"
            )
            continue
        embedded_runtimes.extend(embedded_runtime_sources(tree))

    if not embedded_runtimes:
        failures.append("no embedded DDP runtime source assignment found")
    for runtime_index, runtime_source in enumerate(embedded_runtimes):
        try:
            ast.parse(
                runtime_source,
                filename=f"{path}:embedded_runtime_{runtime_index}",
            )
        except SyntaxError as error:
            failures.append(f"embedded runtime {runtime_index} is invalid: {error}")

    searchable_source = combined_source + "\n" + "\n".join(embedded_runtimes)
    for label, token in REQUIRED_TOKENS.items():
        if token not in searchable_source:
            failures.append(f"missing {label}: {token}")
    for label, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(searchable_source):
            failures.append(f"forbidden {label}: {pattern.pattern}")

    if '"world_size": 2' not in searchable_source:
        failures.append('missing explicit `"world_size": 2` configuration')
    if "n_gpu == 2" not in searchable_source:
        failures.append("missing exact two-GPU assertion")
    if '"T4"' not in searchable_source and "'T4'" not in searchable_source:
        failures.append("missing T4 model assertion")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Kaggle T4 x2 torchrun/DDP notebook structure."
    )
    parser.add_argument("notebooks", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed = False
    for notebook_path in args.notebooks:
        failures = validate_notebook(notebook_path)
        if failures:
            failed = True
            print(f"[FAIL] {notebook_path}", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
        else:
            print(f"[OK] {notebook_path}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
