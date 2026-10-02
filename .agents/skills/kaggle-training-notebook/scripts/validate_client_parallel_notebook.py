#!/usr/bin/env python3
"""Validate two-T4 client-parallel Kaggle notebook structure."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


REQUIRED_TOKENS = {
    "ordinary subprocess launcher": "subprocess.run(",
    "client-parallel script path": "CLIENT_PARALLEL_SCRIPT_PATH",
    "spawn context": 'get_context("spawn")',
    "worker function": "worker_main(",
    "rank-local CUDA binding": "torch.cuda.set_device(gpu_id)",
    "two GPU workers": "range(2)",
    "per-client batch": '"per_client_batch_size": 1024',
    "CUDA AMP autocast": 'autocast("cuda")',
    "CUDA AMP scaler": 'GradScaler("cuda")',
    "CUDA streams": "torch.cuda.Stream",
    "stream candidates": '"stream_candidates": [',
    "CUDA timing events": "torch.cuda.Event",
    "GPU cache fraction": '"gpu_cache_fraction": 0.7',
    "cache budget": "cache_budget_bytes",
    "CPU cache contract validation": "validate_numpy_classification_cache(",
    "workload assignment": "client_assignment",
    "worker error propagation": '"worker_error"',
    "traceback logging": "traceback.format_exc()",
    "best checkpoint": "best.pt",
    "last checkpoint": "last.pt",
    "parent log": "run.log",
    "worker-one compatibility log": "rank_1.log",
    "Kaggle input": "/kaggle/input",
    "Kaggle output": "/kaggle/working",
    "Kaggle temporary storage": "/kaggle/temp",
    "initial states in checkpoint": "initial_model_state_dicts_by_family",
    "initialization hashes": "initialization_hashes",
    "personalized states before fine-tune": (
        "personalized_model_state_dicts_before_finetune"
    ),
}

REQUIRED_OUTPUT_PATHS = (
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "logs/run.log",
    "logs/rank_1.log",
    "metrics/config.json",
    "metrics/dataset_summary.json",
    "metrics/client_class_distribution.csv",
    "metrics/history_round.csv",
    "metrics/history_round.json",
    "metrics/history_client.csv",
    "metrics/history_client.json",
    "metrics/history_local_epoch.csv",
    "metrics/history_local_epoch.json",
    "metrics/summary.json",
    "metrics/classification_report.json",
    "metrics/classification_report.csv",
    "metrics/confusion_matrix.csv",
    "metrics/confusion_matrix.npy",
    "metrics/communication_costs.json",
    "metrics/communication_costs.csv",
    "metrics/runtime_breakdown.json",
    "artifacts/class_distribution.png",
    "artifacts/accuracy_f1_curves.png",
    "artifacts/loss_curves.png",
    "artifacts/confusion_matrix.png",
    "artifacts/per_class_f1.png",
    "artifacts/runtime_per_round.png",
    "artifacts/communication_cumulative.png",
)

FORBIDDEN_PATTERNS = {
    "DistributedDataParallel": re.compile(r"\bDistributedDataParallel\b"),
    "DDP wrapper": re.compile(r"\bDDP\s*\("),
    "torch distributed": re.compile(r"\btorch\.distributed\b"),
    "distributed sampler": re.compile(r"\bDistributedSampler\b"),
    "process group": re.compile(r"\binit_process_group\b"),
    "torchrun": re.compile(r"torch\.distributed\.run|torchrun"),
    "NCCL configuration": re.compile(r"(?<![A-Za-z0-9_])(?:TORCH_)?NCCL_[A-Z_]"),
    "SyncBatchNorm": re.compile(r"\bSyncBatchNorm\b"),
    "DataParallel wrapper": re.compile(r"(?<!Distributed)DataParallel\s*\("),
    "invalid logging grouping": re.compile(r"%(?:\([^)]+\))?[-+ #0]*,"),
    "unresolved TODO": re.compile(r"\bTODO\b"),
    "unresolved confirmed placeholder": re.compile(r"\bCONFIRMED[_-]"),
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


class LoopItemVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loop_depth = 0
        self.locations: list[int] = []

    def visit_For(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node):
        if (
            self.loop_depth
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "item"
        ):
            self.locations.append(getattr(node, "lineno", -1))
        self.generic_visit(node)


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def call_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = call_path(node.value)
        return (*parent, node.attr) if parent else None
    return None


def assignment_is_stream_local(
    function: ast.FunctionDef,
    target_name: str,
    constructor_path: tuple[str, ...],
) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.With):
            continue
        owns_stream = any(
            isinstance(item.context_expr, ast.Call)
            and call_path(item.context_expr.func) == ("torch", "cuda", "stream")
            and len(item.context_expr.args) == 1
            and isinstance(item.context_expr.args[0], ast.Name)
            and item.context_expr.args[0].id == "stream"
            for item in node.items
        )
        if not owns_stream:
            continue
        for child in ast.walk(node):
            if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                continue
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            if not any(
                isinstance(target, ast.Name) and target.id == target_name
                for target in targets
            ):
                continue
            value = child.value
            if isinstance(value, ast.Call) and call_path(value.func) == constructor_path:
                return True
    return False


def contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name
        for child in ast.walk(node)
    )


def has_cudnn_rnn_module(tree: ast.AST) -> bool:
    rnn_constructors = {
        ("torch", "nn", "GRU"),
        ("torch", "nn", "LSTM"),
        ("torch", "nn", "RNN"),
        ("nn", "GRU"),
        ("nn", "LSTM"),
        ("nn", "RNN"),
    }
    return any(
        isinstance(node, ast.Call) and call_path(node.func) in rnn_constructors
        for node in ast.walk(tree)
    )


def validate_proxy_backward_model_mode(tree: ast.AST) -> list[str]:
    """Reject eval-mode cuDNN RNN forwards traversed by proxy backward."""
    if not has_cudnn_rnn_module(tree):
        return []

    failures = []
    for function_name in ("train_client", "benchmark_family"):
        function = find_function(tree, function_name)
        if function is None:
            continue
        proxy_backward_lines = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward"
            and contains_name(node, "proxy_loss")
        ]
        if not proxy_backward_lines:
            continue
        first_backward_line = min(proxy_backward_lines)
        mode_calls = [
            (node.lineno, node.func.attr)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr in {"train", "eval"}
            and node.lineno < first_backward_line
        ]
        if not mode_calls or max(mode_calls)[1] != "train":
            observed = max(mode_calls)[1] if mode_calls else "unset"
            failures.append(
                f"{function_name}: local model mode before proxy backward is "
                f"{observed}; a frozen cuDNN GRU/LSTM/RNN must remain in train "
                "mode when backward traverses it to update an upstream proxy"
            )
    return failures


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
        failures.append("no embedded client-parallel runtime source found")
    for runtime_index, runtime_source in enumerate(embedded_runtimes):
        try:
            runtime_tree = ast.parse(
                runtime_source,
                filename=f"{path}:embedded_runtime_{runtime_index}",
            )
        except SyntaxError as error:
            failures.append(f"embedded runtime {runtime_index} is invalid: {error}")
            continue
        visitor = LoopItemVisitor()
        visitor.visit(runtime_tree)
        if visitor.locations:
            failures.append(
                "per-batch/per-loop .item() calls found at embedded runtime lines "
                + ", ".join(map(str, visitor.locations))
            )
        context_factory = find_function(runtime_tree, "make_training_context")
        if context_factory is None:
            failures.append("embedded runtime is missing make_training_context")
        else:
            stream_local_tensors = {
                "order": ("torch", "randperm"),
                "loss_sums": ("torch", "zeros"),
            }
            for tensor_name, constructor in stream_local_tensors.items():
                if not assignment_is_stream_local(
                    context_factory,
                    tensor_name,
                    constructor,
                ):
                    failures.append(
                        f"{tensor_name} must be created inside its owning "
                        "with torch.cuda.stream(stream) context"
                    )
        failures.extend(validate_proxy_backward_model_mode(runtime_tree))

    searchable_source = combined_source + "\n" + "\n".join(embedded_runtimes)
    for label, token in REQUIRED_TOKENS.items():
        if token not in searchable_source:
            failures.append(f"missing {label}: {token}")
    for relative_path in REQUIRED_OUTPUT_PATHS:
        if relative_path not in searchable_source:
            failures.append(f"missing required output path: {relative_path}")
    for label, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(searchable_source):
            failures.append(f"forbidden {label}: {pattern.pattern}")

    if '"execution_mode": "client_parallel"' not in searchable_source:
        failures.append("missing explicit client_parallel execution mode")
    if "n_gpu == 2" not in searchable_source:
        failures.append("missing exact two-GPU assertion")
    if '"T4"' not in searchable_source and "'T4'" not in searchable_source:
        failures.append("missing T4 model assertion")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Kaggle T4 x2 client-parallel notebook structure."
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
