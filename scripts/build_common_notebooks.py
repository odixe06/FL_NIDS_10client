from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = Path(__file__).with_name("common_client_parallel_runtime.py")

FEATURE_COLUMNS = [
    "ack_flag_number",
    "AVG",
    "Std",
    "UDP",
    "fin_count",
    "Max",
    "TCP",
    "syn_count",
    "Protocol Type",
    "Rate",
    "IAT",
    "syn_flag_number",
    "rst_flag_number",
    "Tot sum",
    "HTTPS",
    "ack_count",
    "fin_flag_number",
    "HTTP",
    "rst_count",
    "Header_Length",
    "psh_flag_number",
    "ICMP",
    "Time_To_Live",
    "ARP",
    "DNS",
]

FAMILIES = {
    "gru": {"title": "GRU", "parameters": 40034},
    "transformer": {"title": "Transformer", "parameters": 71010},
    "cnn1d": {"title": "CNN-1D", "parameters": 35874},
}

METHODS = {
    "fd_ids_noniid": {
        "display": "FD-IDS",
        "evaluation_scope": "server_only",
        "server_classifier_available": True,
        "server_state_type": "global_classifier",
        "persistent_personalized_clients": False,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "split_policy": "100_percent_local_train_no_validation",
    },
    "perfed_skd": {
        "display": "PerFed-SKD",
        "evaluation_scope": "server_and_clients",
        "server_classifier_available": True,
        "server_state_type": "global_classifier",
        "persistent_personalized_clients": True,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "split_policy": "deterministic_stratified_90_train_10_validation",
    },
    "pfedes": {
        "display": "pFedES",
        "evaluation_scope": "clients_only",
        "server_classifier_available": False,
        "server_state_type": "global_proxy_feature_extractor",
        "server_evaluation_not_applicable_reason": (
            "The pFedES server state is a 25-to-25 proxy feature extractor and "
            "does not return 34-class logits."
        ),
        "persistent_personalized_clients": True,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "split_policy": "100_percent_local_train_no_validation",
    },
    "proxymodel": {
        "display": "ProxyModel",
        "evaluation_scope": "server_and_clients",
        "server_classifier_available": True,
        "server_state_type": "global_cnn1d_proxy_classifier",
        "persistent_personalized_clients": True,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "split_policy": "100_percent_local_train_no_validation",
    },
}


def source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def make_config(method: str, family: str) -> dict:
    method_config = METHODS[method]
    run_name = f"task10_{method}_10c_{family}"
    config = {
        "method": method,
        "method_name": method,
        "scenario": f"10_clients_{family}",
        "run_name": run_name,
        "model_family": family,
        "execution_mode": "client_parallel",
        "input_dir": "/kaggle/input/datasets/odixe0502/data-for-10clients",
        "output_dir": f"/kaggle/working/{run_name}",
        "temp_dir": f"/kaggle/temp/{run_name}_runtime",
        "cache_dir": f"/kaggle/temp/{run_name}_cache",
        "manifest_path": f"/kaggle/temp/{run_name}_manifest.json",
        "num_clients": 10,
        "num_features": 25,
        "num_classes": 34,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": "Label",
        "rounds": 10,
        "communication_rounds": 10,
        "local_epochs": 1,
        "per_client_batch_size": 1024,
        # Evaluation has no optimizer/activation-gradient state. A larger batch
        # keeps the small 34-class models busy while remaining conservative on
        # a 16 GiB T4 even with the 70% dataset-cache ceiling.
        "evaluation_batch_size": 32768,
        "csv_chunk_rows": 262144,
        "gpu_copy_chunk_rows": 262144,
        "dataloader_workers_per_gpu": 2,
        "optimizer": method_config["optimizer"],
        "learning_rate": method_config["learning_rate"],
        "momentum": 0.0,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "scheduler": None,
        "seed": 42,
        "initialization_seed": 42,
        "gpu_cache_fraction": 0.7,
        "stream_candidates": [1, 2],
        "stream_min_speedup": 1.10,
        "stream_benchmark_warmup_steps": 3,
        "stream_benchmark_steps": 12,
        "worker_count": 2,
        "worker_timeout_seconds": 86400,
        "mixed_precision": "cuda_amp",
        "expected_parameter_count": FAMILIES[family]["parameters"],
        "local_train_fraction": 0.9 if method == "perfed_skd" else 1.0,
        "local_validation_fraction": 0.1 if method == "perfed_skd" else 0.0,
        "create_local_validation": method == "perfed_skd",
        "client_selection_metric": "accuracy" if method == "perfed_skd" else None,
        "client_selection_source": (
            "local_validation" if method == "perfed_skd" else None
        ),
        "split_policy": method_config["split_policy"],
        "evaluate_global_test_each_round": True,
        "global_test_controls_training": False,
        "create_best_checkpoint": False,
        "comparison_round": 10,
        "comparison_metrics": [
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
        ],
        "evaluation_scope": method_config["evaluation_scope"],
        "server_classifier_available": method_config["server_classifier_available"],
        "server_state_type": method_config["server_state_type"],
        "server_evaluation_not_applicable_reason": method_config.get(
            "server_evaluation_not_applicable_reason", ""
        ),
        "persistent_personalized_clients": method_config[
            "persistent_personalized_clients"
        ],
        "temperature": 3.0 if method == "fd_ids_noniid" else 1.0,
        "fedprox_mu": 0.01,
        "hard_loss_weight": 0.5,
        "proximal_weight": 0.1,
        "distillation_lambda": 1.0,
        "adaptive_epsilon": 1e-8,
        "enhanced_loss_weight": 0.1,
        "final_personalized_finetune_epochs": 1 if method == "proxymodel" else 0,
    }
    return config


def expected_outputs(method: str) -> list[str]:
    outputs = [
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
    ]
    if method == "perfed_skd":
        outputs.extend(
            ["metrics/validation_metrics.csv", "metrics/validation_metrics.json"]
        )
    return outputs


def introduction(method: str, family: str) -> str:
    method_config = METHODS[method]
    split = (
        "90% local train / 10% local validation; validation accuracy alone "
        "selects next-round clients"
        if method == "perfed_skd"
        else "100% of every local client file is used for training"
    )
    return f"""# Task10 — {method_config['display']} — 10 client {FAMILIES[family]['title']}

Notebook tái xây dựng phương pháp **{method_config['display']}** trên CICIoT2023
đã tiền xử lý: 25 feature, 34 lớp và 10 client non-IID. Chính sách local data:
{split}.

Sau mỗi trong 10 round, notebook lưu toàn bộ checkpoint áp dụng và đánh giá
đúng scope server/client trên toàn bộ `global_test_data.csv` bằng 10 metrics:
accuracy; macro/micro/weighted precision; macro/micro/weighted recall; và
macro/micro/weighted F1. Global test không điều khiển huấn luyện hoặc lựa chọn
checkpoint. Không tạo checkpoint tốt nhất; round 10 là endpoint so sánh cố định.

Runtime dùng CPU coordinator và đúng hai persistent GPU worker, mỗi worker gắn
với một NVIDIA T4; batch 1024 là batch của một client update trên một GPU.
"""


def build_notebook(method: str, family: str, runtime_source: str) -> dict:
    config = make_config(method, family)
    outputs = expected_outputs(method)
    environment_cell = """import platform
import torch

assert torch.cuda.is_available(), "CUDA is required"
n_gpu = torch.cuda.device_count()
assert n_gpu == 2, f"Expected exactly two GPUs, found {n_gpu}"
gpu_names = [torch.cuda.get_device_name(index) for index in range(n_gpu)]
assert all("T4" in name for name in gpu_names), f"Expected NVIDIA T4 x2, found {gpu_names}"
print({
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": gpu_names,
})
"""
    config_cell = (
        "import json\n"
        "from pathlib import Path\n\n"
        f"CONFIG = json.loads(r'''{json.dumps(config, ensure_ascii=False)}''')\n"
        f"EXPECTED_OUTPUTS = {repr(outputs)}\n"
        "assert json.loads(json.dumps(CONFIG)) == CONFIG\n"
        "Path(CONFIG['temp_dir']).mkdir(parents=True, exist_ok=True)\n"
        "Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True)\n"
        "Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)\n"
    )
    runtime_cell = (
        f"RUNTIME_SOURCE = {runtime_source!r}\n"
        "CLIENT_PARALLEL_SCRIPT_PATH = Path(CONFIG['temp_dir']) / 'training_entry.py'\n"
        "CLIENT_PARALLEL_SCRIPT_PATH.write_text(RUNTIME_SOURCE, encoding='utf-8')\n"
        "compile(RUNTIME_SOURCE, str(CLIENT_PARALLEL_SCRIPT_PATH), 'exec')\n"
        "print(f'Wrote runtime: {CLIENT_PARALLEL_SCRIPT_PATH}')\n"
    )
    preprocessing_cell = """import importlib.util
import logging

spec = importlib.util.spec_from_file_location("task10_runtime", CLIENT_PARALLEL_SCRIPT_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
runtime.create_output_dirs(CONFIG)
notebook_logger = runtime.configure_logger(Path(CONFIG["output_dir"]) / "logs" / "run.log", "notebook_preprocess")
datasets = runtime.prepare_datasets(CONFIG, notebook_logger)
runtime.write_dataset_outputs(Path(CONFIG["output_dir"]), datasets, CONFIG)
MANIFEST_PATH = Path(CONFIG["manifest_path"])
runtime.atomic_json(MANIFEST_PATH, {"config": CONFIG, "datasets": datasets})
print(f"Prepared manifest: {MANIFEST_PATH}")
"""
    launch_cell = """import os
import subprocess
import sys

for handler in list(logging.getLogger("notebook_preprocess").handlers):
    handler.close()
    logging.getLogger("notebook_preprocess").removeHandler(handler)
launch_environment = os.environ.copy()
launch_environment.update({
    "TRAINING_MANIFEST": str(MANIFEST_PATH),
    "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg",
})
subprocess.run([sys.executable, str(CLIENT_PARALLEL_SCRIPT_PATH)], check=True, env=launch_environment)
"""
    verify_cell = """import pandas as pd
from IPython.display import display
from PIL import Image

output_dir = Path(CONFIG["output_dir"])
missing = []
for relative in EXPECTED_OUTPUTS:
    path = output_dir / relative
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        missing.append(relative)
if missing:
    raise RuntimeError(f"Missing or empty outputs: {missing}")
metrics = pd.read_csv(output_dir / "metrics" / "evaluation_metrics.csv")
runtime.verify_evaluation_metrics(metrics.to_dict("records"), CONFIG, datasets["global_test"]["rows"])
checkpoint_manifest = json.loads((output_dir / "checkpoints" / "checkpoint_manifest.json").read_text(encoding="utf-8"))
runtime.verify_checkpoint_manifest(output_dir, CONFIG, checkpoint_manifest["entries"])
summary = json.loads((output_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "complete"
display(metrics.tail(20))
for filename in [
    "class_distribution.png",
    "evaluation_metric_curves.png",
    "loss_curves.png",
    "runtime_per_round.png",
    "communication_cumulative.png",
    "gpu_utilization.png",
]:
    display(Image.open(output_dir / "artifacts" / filename))
"""
    return {
        "cells": [
            markdown_cell(introduction(method, family)),
            code_cell(environment_cell),
            code_cell(config_cell),
            markdown_cell("## Embedded client-parallel training runtime\n"),
            code_cell(runtime_cell),
            markdown_cell("## Validate and cache the confirmed dataset\n"),
            code_cell(preprocessing_cell),
            markdown_cell("## Launch the two persistent T4 workers\n"),
            code_cell(launch_cell),
            markdown_cell("## Verify outputs and render diagnostics\n"),
            code_cell(verify_cell),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(method_filter: str | None = None) -> None:
    runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
    for method in METHODS:
        if method_filter is not None and method != method_filter:
            continue
        for family in FAMILIES:
            folder = ROOT / method / f"10_clients_{family}"
            folder.mkdir(parents=True, exist_ok=True)
            notebook_path = folder / f"{method}_10_clients_{family}.ipynb"
            notebook = build_notebook(method, family, runtime_source)
            notebook_path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False),
                encoding="utf-8",
            )
            print(notebook_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build task10 client-parallel notebooks")
    parser.add_argument("--method", choices=sorted(METHODS), default=None)
    arguments = parser.parse_args()
    main(arguments.method)
