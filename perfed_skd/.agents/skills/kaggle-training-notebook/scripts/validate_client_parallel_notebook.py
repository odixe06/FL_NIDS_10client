#!/usr/bin/env python3
"""Structurally validate a locked two-T4 federated benchmark notebook."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "micro_precision",
    "weighted_precision",
    "macro_recall",
    "micro_recall",
    "weighted_recall",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
)

REQUIRED_TOKENS = {
    "ordinary subprocess launcher": "subprocess.run(",
    "client-parallel script path": "CLIENT_PARALLEL_SCRIPT_PATH",
    "spawn context": 'get_context("spawn")',
    "worker function": "worker_main(",
    "rank-local CUDA binding": "torch.cuda.set_device(gpu_id)",
    "two persistent GPU workers": "range(2)",
    "CUDA AMP autocast": 'autocast("cuda")',
    "CUDA AMP scaler": 'GradScaler("cuda")',
    "CUDA streams": "torch.cuda.Stream",
    "stream candidates": '"stream_candidates": [',
    "CUDA timing events": "torch.cuda.Event",
    "cache budget": "cache_budget_bytes",
    "CPU cache contract validation": "validate_numpy_classification_cache(",
    "workload assignment": "client_assignment",
    "worker error propagation": '"worker_error"',
    "traceback logging": "traceback.format_exc()",
    "checkpoint manifest verification": "verify_checkpoint_manifest(",
    "evaluation metric verification": "verify_evaluation_metrics(",
    "checkpoint manifest": "checkpoint_manifest.json",
    "round checkpoint folder": "round_",
    "server checkpoint": "server.pt",
    "client checkpoint": "client_",
    "parent log": "run.log",
    "worker zero log": "worker_0.log",
    "worker one log": "worker_1.log",
    "global test file": "global_test_data.csv",
    "Kaggle input": "/kaggle/input",
    "Kaggle output": "/kaggle/working",
    "Kaggle temporary storage": "/kaggle/temp",
}

REQUIRED_OUTPUT_PATHS = (
    "checkpoints/checkpoint_manifest.json",
    "logs/run.log",
    "logs/worker_0.log",
    "logs/worker_1.log",
    "metrics/config.json",
    "metrics/dataset_summary.json",
    "metrics/client_class_distribution.csv",
    "metrics/evaluation_metrics.csv",
    "metrics/evaluation_metrics.json",
    "metrics/final_round_metrics.csv",
    "metrics/final_round_metrics.json",
    "metrics/server_evaluation_status.json",
    "metrics/summary.json",
    "metrics/communication_costs.json",
    "metrics/runtime_breakdown.json",
    "metrics/confusion_matrices",
    "artifacts/class_distribution.png",
    "artifacts/evaluation_metric_curves.png",
    "artifacts/loss_curves.png",
    "artifacts/runtime_per_round.png",
    "artifacts/communication_cumulative.png",
    "artifacts/gpu_utilization.png",
)

FORBIDDEN_PATTERNS = {
    "best checkpoint": re.compile(r"best\.pt"),
    "DistributedDataParallel": re.compile(r"\bDistributedDataParallel\b"),
    "DDP wrapper": re.compile(r"\bDDP\s*\("),
    "torch distributed": re.compile(r"\btorch\.distributed\b"),
    "distributed sampler": re.compile(r"\bDistributedSampler\b"),
    "process group": re.compile(r"\binit_process_group\b"),
    "torchrun": re.compile(r"torch\.distributed\.run|torchrun"),
    "NCCL configuration": re.compile(r"(?<![A-Za-z0-9_])(?:TORCH_)?NCCL_[A-Z_]"),
    "SyncBatchNorm": re.compile(r"\bSyncBatchNorm\b"),
    "DataParallel wrapper": re.compile(r"(?<!Distributed)DataParallel\s*\("),
    "early stopping": re.compile(r"\bearly_stopping\b"),
    "invalid logging grouping": re.compile(r"%(?:\([^)]+\))?[-+ #0]*,"),
    "read-only NumPy view passed to torch.from_numpy": re.compile(
        r"torch\.from_numpy\s*\(\s*np\.asarray\s*\("
    ),
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
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def has_cudnn_rnn_module(tree: ast.AST) -> bool:
    constructors = {
        ("torch", "nn", "GRU"),
        ("torch", "nn", "LSTM"),
        ("torch", "nn", "RNN"),
        ("nn", "GRU"),
        ("nn", "LSTM"),
        ("nn", "RNN"),
    }
    return any(
        isinstance(node, ast.Call) and call_path(node.func) in constructors
        for node in ast.walk(tree)
    )


def logger_info_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and call_path(node.func) == ("logger", "info")
    )


def validate_worker_diagnostic_logging(tree: ast.AST) -> list[str]:
    worker = find_function(tree, "worker_main")
    if worker is None:
        return ["embedded runtime is missing worker_main"]

    task_loops = [node for node in ast.walk(worker) if isinstance(node, ast.While)]
    if not task_loops:
        return ["worker_main is missing its persistent task loop"]
    first_task_line = min(node.lineno for node in task_loops)

    startup_logged = False
    for node in ast.walk(worker):
        if not logger_info_call(node) or node.lineno >= first_task_line:
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            message = str(node.args[0].value).lower()
            if "worker" in message and "start" in message:
                startup_logged = True
                break

    task_logged = any(
        logger_info_call(node)
        and contains_name(node, "task_id")
        and contains_name(node, "kind")
        for loop in task_loops
        for node in ast.walk(loop)
    )

    failures = []
    if not startup_logged:
        failures.append(
            "worker_main must emit a startup INFO record before its task loop; "
            "creating a log handler alone leaves a zero-byte worker log"
        )
    if not task_logged:
        failures.append(
            "worker_main must emit task lifecycle INFO records containing "
            "task_id and kind"
        )
    return failures


def validate_numpy_bridge_writability(tree: ast.AST) -> list[str]:
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if call_path(node.func) != ("torch", "from_numpy") or not node.args:
            continue
        source = node.args[0]
        if isinstance(source, ast.Call) and call_path(source.func) == ("np", "asarray"):
            failures.append(
                f"line {node.lineno}: np.asarray may preserve a read-only "
                "memmap view; use bounded np.array(..., copy=True) staging "
                "before torch.from_numpy"
            )
        if isinstance(source, ast.Call) and call_path(source.func) == ("np", "array"):
            copy_keywords = [
                keyword.value
                for keyword in source.keywords
                if keyword.arg == "copy"
            ]
            if copy_keywords and not all(
                isinstance(value, ast.Constant) and value.value is True
                for value in copy_keywords
            ):
                failures.append(
                    f"line {node.lineno}: np.array passed to torch.from_numpy "
                    "must use copy=True when staging a possible read-only memmap"
                )
    return failures


def validate_proxy_backward_model_mode(tree: ast.AST) -> list[str]:
    """Reject eval-mode cuDNN RNN forwards traversed by proxy backward."""
    if not has_cudnn_rnn_module(tree):
        return []

    failures = []
    for function_name in ("train_client", "benchmark_family"):
        function = find_function(tree, function_name)
        if function is None:
            continue
        backward_lines = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward"
            and contains_name(node, "proxy_loss")
        ]
        if not backward_lines:
            continue
        first_backward = min(backward_lines)
        mode_calls = [
            (node.lineno, node.func.attr)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr in {"train", "eval"}
            and node.lineno < first_backward
        ]
        if not mode_calls or max(mode_calls)[1] != "train":
            observed = max(mode_calls)[1] if mode_calls else "unset"
            failures.append(
                f"{function_name}: local model mode before proxy backward is "
                f"{observed}; a frozen cuDNN recurrent module must remain in "
                "train mode when backward traverses it"
            )
    return failures


def require_one_of(source: str, label: str, alternatives: tuple[str, ...]) -> str | None:
    if any(value in source for value in alternatives):
        return None
    return f"missing {label}; expected one of: {', '.join(alternatives)}"


def validate_notebook(path: Path) -> list[str]:
    failures: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"invalid notebook JSON: {error}"]

    if notebook.get("nbformat") != 4:
        failures.append("nbformat must be 4")

    cells = list(code_cells(notebook))
    combined_source = "\n".join(source for _, source in cells)
    embedded_runtimes: list[str] = []

    for cell_index, source in cells:
        if source.lstrip().startswith(("!", "%")):
            continue
        try:
            tree = ast.parse(source, filename=f"{path}:cell_{cell_index}")
        except SyntaxError as error:
            failures.append(f"cell {cell_index} is not valid Python: {error}")
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
                "per-loop .item() calls found at embedded runtime lines "
                + ", ".join(map(str, visitor.locations))
            )

        context_factory = find_function(runtime_tree, "make_training_context")
        if context_factory is None:
            failures.append("embedded runtime is missing make_training_context")
        else:
            for tensor_name, constructor in {
                "order": ("torch", "randperm"),
                "loss_sums": ("torch", "zeros"),
            }.items():
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
        failures.extend(validate_worker_diagnostic_logging(runtime_tree))
        failures.extend(validate_numpy_bridge_writability(runtime_tree))

    searchable_source = combined_source + "\n" + "\n".join(embedded_runtimes)

    for label, token in REQUIRED_TOKENS.items():
        if token not in searchable_source:
            failures.append(f"missing {label}: {token}")
    for metric_name in METRIC_NAMES:
        if metric_name not in searchable_source:
            failures.append(f"missing required metric name: {metric_name}")
    for relative_path in REQUIRED_OUTPUT_PATHS:
        if relative_path not in searchable_source:
            failures.append(f"missing required output path: {relative_path}")
    for label, pattern in FORBIDDEN_PATTERNS.items():
        if pattern.search(searchable_source):
            failures.append(f"forbidden {label}: {pattern.pattern}")

    required_config_values = (
        ("client_parallel execution mode", ('"execution_mode": "client_parallel"',)),
        ("ten clients", ('"num_clients": 10',)),
        ("ten rounds", ('"rounds": 10', '"communication_rounds": 10')),
        ("per-client batch 1024", ('"per_client_batch_size": 1024',)),
        (
            "per-round global-test evaluation",
            (
                '"evaluate_global_test_each_round": true',
                '"evaluate_global_test_each_round": True',
            ),
        ),
        (
            "disabled best checkpoint",
            ('"create_best_checkpoint": false', '"create_best_checkpoint": False'),
        ),
        ("GPU cache fraction 0.7", ('"gpu_cache_fraction": 0.7',)),
    )
    for label, alternatives in required_config_values:
        failure = require_one_of(searchable_source, label, alternatives)
        if failure:
            failures.append(failure)

    is_perfed_skd = any(
        token in searchable_source
        for token in (
            '"method": "perfed_skd"',
            '"method_name": "perfed_skd"',
            '"method": "PerFed-SKD"',
        )
    )
    if is_perfed_skd:
        perfed_requirements = (
            ("PerFed-SKD 90% local train", ('"local_train_fraction": 0.9',)),
            (
                "PerFed-SKD 10% local validation",
                ('"local_validation_fraction": 0.1',),
            ),
            (
                "enabled PerFed-SKD local validation",
                (
                    '"create_local_validation": true',
                    '"create_local_validation": True',
                ),
            ),
            (
                "validation accuracy client selection",
                ('"client_selection_metric": "accuracy"',),
            ),
            (
                "validation-only client selection source",
                ('"client_selection_source": "local_validation"',),
            ),
        )
        for label, alternatives in perfed_requirements:
            failure = require_one_of(searchable_source, label, alternatives)
            if failure:
                failures.append(failure)
        for relative_path in (
            "metrics/validation_metrics.csv",
            "metrics/validation_metrics.json",
        ):
            if relative_path not in searchable_source:
                failures.append(
                    f"missing PerFed-SKD validation output path: {relative_path}"
                )
        if "validate_client_split_accounting(" not in searchable_source:
            failures.append("missing exhaustive PerFed-SKD split validation")
    else:
        full_local_requirements = (
            ("100% local training", ('"local_train_fraction": 1.0',)),
            (
                "disabled local validation",
                (
                    '"create_local_validation": false',
                    '"create_local_validation": False',
                ),
            ),
        )
        for label, alternatives in full_local_requirements:
            failure = require_one_of(searchable_source, label, alternatives)
            if failure:
                failures.append(failure)
        is_fedcaps = any(
            token in searchable_source
            for token in (
                '"method": "fedcaps"',
                '"method_name": "fedcaps"',
                '"method": "FedCAPS"',
            )
        )
        if is_fedcaps:
            fedcaps_requirements = (
                (
                    "FedCAPS five-fold search CV",
                    ('"fedcaps_search_cv_folds": 5',),
                ),
                (
                    "FedCAPS 100,000-row client search-pool cap",
                    ('"fedcaps_search_pool_max_rows_per_client": 100000',),
                ),
            )
            for label, alternatives in fedcaps_requirements:
                failure = require_one_of(searchable_source, label, alternatives)
                if failure:
                    failures.append(failure)

    if "n_gpu == 2" not in searchable_source:
        failures.append("missing exact two-GPU assertion")
    if '"T4"' not in searchable_source and "'T4'" not in searchable_source:
        failures.append("missing T4 model assertion")

    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a Kaggle T4 x2 client-parallel benchmark notebook."
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
