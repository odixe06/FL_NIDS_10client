#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "fedcaps_client_parallel_runtime.py"

SCENARIOS = [
    {
        "folder": "10_clients_gru",
        "notebook": "fedcaps_10_clients_gru.ipynb",
        "run_name": "task10_fedcaps_10c_gru",
        "scenario_name": "10 clients GRU",
        "model_family": "gru",
    },
    {
        "folder": "10_clients_transformer",
        "notebook": "fedcaps_10_clients_transformer.ipynb",
        "run_name": "task10_fedcaps_10c_transformer",
        "scenario_name": "10 clients Transformer",
        "model_family": "transformer",
    },
    {
        "folder": "10_clients_cnn1d",
        "notebook": "fedcaps_10_clients_cnn1d.ipynb",
        "run_name": "task10_fedcaps_10c_cnn1d",
        "scenario_name": "10 clients CNN-1D",
        "model_family": "cnn1d",
    },
]


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def notebook_for(scenario: dict, runtime_source: str) -> dict:
    config = {
        "run_name": scenario["run_name"],
        "scenario": scenario["folder"],
        "scenario_name": scenario["scenario_name"],
        "method": "fedcaps",
        "execution_mode": "client_parallel",
        "input_dir": "/kaggle/input/datasets/odixe0502/data-for-10clients",
        "output_dir": f"/kaggle/working/{scenario['run_name']}",
        "temporary_dir": f"/kaggle/temp/{scenario['run_name']}_cache",
        "model_family": scenario["model_family"],
        "client_models": {str(client_id): scenario["model_family"] for client_id in range(1, 11)},
        "num_clients": 10,
        "num_features": 25,
        "num_classes": 34,
        "seed": 42,
        "initialization_seed": 42,
        "rounds": 10,
        "communication_rounds": 10,
        "local_epochs_per_round": 1,
        "local_train_fraction": 1.0,
        "local_validation_fraction": 0.0,
        "create_local_validation": False,
        "evaluate_global_test_each_round": True,
        "create_best_checkpoint": False,
        "persistent_personalized_clients": True,
        "server_is_classifier": False,
        "evaluation_scope": "clients_only",
        "fedcaps_search_cv_folds": 5,
        "fedcaps_search_pool_max_rows_per_client": 100000,
        "marlfs_epochs": 300,
        "marlfs_log_interval": 10,
        "record_permutations": 25,
        "encoder_isab_layers": 2,
        "encoder_attention_heads": 4,
        "embedding_dimension": 128,
        "inducing_points": 32,
        "pma_seed_vectors": 32,
        "encoder_batch_size": 64,
        "encoder_learning_rate": 0.001,
        "encoder_max_epochs": 100,
        "encoder_patience": 10,
        "permutation_invariance_tolerance": 1e-5,
        "ppo_seed_count": 25,
        "ppo_search_epochs": 10,
        "ppo_batch_size": 512,
        "actor_learning_rate": 0.0003,
        "critic_learning_rate": 0.001,
        "reward_lambda": 0.1,
        "ppo_discount_factor": 0.99,
        "ppo_steps_per_epoch": 1000,
        "ppo_clip_ratio": 0.2,
        "ppo_feedback_interval": 100,
        "ppo_update_epochs": 4,
        "ppo_entropy_coefficient": 0.001,
        "actor_step_scale": 0.1,
        "critic_pretrain_epochs": 200,
        "critic_calibration_steps": 10,
        "minimum_subset_size": 2,
        "candidate_evaluator_epochs": 1,
        "classifier_optimizer": "SGD",
        "classifier_learning_rate": 0.01,
        "per_client_batch_size": 1024,
        "gradient_accumulation_steps": 1,
        "scheduler": None,
        "loss": "cross_entropy",
        "class_weighting": False,
        "mixed_precision": True,
        "gpu_cache_fraction": 0.7,
        "stream_candidates": [1, 2],
        "stream_improvement_margin": 1.0,
        "workload_assignment_policy": "phase_specific_exhaustive_bipartition",
        "search_parallelism_unit": "independent_stratified_cv_fold",
        "require_multistream_deterministic_replay": True,
        "persistent_global_test_gpu_cache": True,
        "gpu_monitor_interval_seconds": 5.0,
        "csv_chunksize": 250000,
        "gpu_copy_chunk_rows": 250000,
        "worker_timeout_seconds": 43200,
        "common_metrics": [
            "accuracy", "macro_precision", "micro_precision", "weighted_precision",
            "macro_recall", "micro_recall", "weighted_recall",
            "macro_f1", "micro_f1", "weighted_f1",
        ],
    }
    title = f"""# FedCAPS on CICIoT2023 — {scenario['scenario_name']}

Notebook này tái triển khai phương pháp FedCAPS của `2510.05535v3` trên dữ liệu
10 client CICIoT2023. Pipeline gồm MARLFS record collection, tăng cường hoán vị,
encoder hai ISAB, decoder PMA–MAB–rFF, PPO search với sample-aware client
feedback, rồi huấn luyện/evaluate classifier cá nhân hóa.

Runtime dùng scheduler riêng cho từng pha: MARLFS/PPO cân theo tổng số bước
5-fold search, final training cân theo toàn bộ local rows, còn personalized
evaluation cân theo số model-test. Mỗi GPU benchmark 1-vs-2 CUDA stream và chỉ
chạy hai fold đồng thời khi throughput tăng và deterministic replay khớp.

Feature subset cuối cùng có độ dài do FedCAPS tự tìm. Notebook yêu cầu Kaggle
accelerator **GPU T4 x2** và ghi toàn bộ output vào
`/kaggle/working/{scenario['run_name']}`.

## Công thức chính

$$W_c=|D_c|/\\sum_j|D_j|,\\qquad
\\hat v(f)=\\sum_cW_cM_c(X_c[f]).$$

$$ISAB_M(f)=MAB(f,H,H),\\qquad H=MAB(I,f,f).$$

$$L_{{rec}}=-\\log P_\\psi(f|E).$$

$$R=\\lambda(\\hat v(f^+)-\\hat v(f))+(1-\\lambda)(1-|f^+|/25).$$

$$L_{{critic}}=T^{{-1}}\\sum_t(V(s_t)-G_t)^2,$$

$$L_{{actor}}=\\hat E_t[\\min(r_tA_t,
\\operatorname{{clip}}(r_t,1-\\epsilon,1+\\epsilon)A_t)].$$
"""
    environment_cell = """from pathlib import Path
import json
import os
import subprocess
import sys
import time

import torch

n_gpu = torch.cuda.device_count()
assert n_gpu == 2, f"Notebook requires exactly two GPUs, observed {n_gpu}"
GPU_NAMES = [torch.cuda.get_device_name(gpu_id) for gpu_id in range(2)]
assert all("T4" in name for name in GPU_NAMES), f"Expected T4 x2, observed {GPU_NAMES}"
print({
    "python": sys.version,
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": GPU_NAMES,
    "vram_gib": [round(torch.cuda.get_device_properties(gpu_id).total_memory / 2**30, 2) for gpu_id in range(2)],
})
"""
    config_cell = "CONFIG = json.loads(r'''" + json.dumps(config, indent=4, ensure_ascii=False) + "''')\n\n" + """assert CONFIG["execution_mode"] == "client_parallel"
assert CONFIG["per_client_batch_size"] == 1024
assert CONFIG["gpu_cache_fraction"] == 0.7
assert CONFIG["stream_candidates"] == [1, 2]
assert CONFIG["workload_assignment_policy"] == "phase_specific_exhaustive_bipartition"
assert CONFIG["search_parallelism_unit"] == "independent_stratified_cv_fold"
assert CONFIG["require_multistream_deterministic_replay"] is True
assert CONFIG["persistent_global_test_gpu_cache"] is True
assert CONFIG["rounds"] == 10
assert CONFIG["local_train_fraction"] == 1.0
assert CONFIG["local_validation_fraction"] == 0.0
assert CONFIG["fedcaps_search_cv_folds"] == 5
assert CONFIG["fedcaps_search_pool_max_rows_per_client"] == 100000
Path(CONFIG["temporary_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
"""
    runtime_cell = "CLIENT_PARALLEL_SCRIPT_SOURCE = " + repr(runtime_source) + "\n\n" + """CLIENT_PARALLEL_SCRIPT_PATH = Path("/kaggle/temp") / f"{CONFIG['run_name']}_client_parallel.py"
CLIENT_PARALLEL_SCRIPT_PATH.write_text(CLIENT_PARALLEL_SCRIPT_SOURCE, encoding="utf-8")
compile(CLIENT_PARALLEL_SCRIPT_SOURCE, str(CLIENT_PARALLEL_SCRIPT_PATH), "exec")
print(f"Wrote self-contained runtime: {CLIENT_PARALLEL_SCRIPT_PATH}")
"""
    manifest_cell = """MANIFEST_PATH = Path("/kaggle/temp") / f"{CONFIG['run_name']}_manifest.json"
MANIFEST = {
    "config": CONFIG,
    "entry_point": str(CLIENT_PARALLEL_SCRIPT_PATH),
    "created_unix_seconds": time.time(),
}
MANIFEST_PATH.write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote manifest: {MANIFEST_PATH}")
"""
    launch_cell = """launch_environment = os.environ.copy()
launch_environment.update({
    "TRAINING_MANIFEST": str(MANIFEST_PATH),
    "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg",
})
started = time.perf_counter()
print("Starting FedCAPS subprocess. Coordinator and worker progress logs will stream below.", flush=True)
completed = subprocess.run(
    [sys.executable, str(CLIENT_PARALLEL_SCRIPT_PATH)],
    check=True,
    env=launch_environment,
    text=True,
)
print({"returncode": completed.returncode, "subprocess_wall_seconds": time.perf_counter() - started})
"""
    verification_cell = """from IPython.display import Image, display
import importlib.util
import pandas as pd

runtime_spec = importlib.util.spec_from_file_location("fedcaps_runtime_verifier", CLIENT_PARALLEL_SCRIPT_PATH)
runtime_module = importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime_module)

RUN_DIR = Path(CONFIG["output_dir"])
CONFUSION_MATRIX_DIR = RUN_DIR / "metrics/confusion_matrices"
required = [
    "checkpoints/checkpoint_manifest.json", "logs/run.log", "logs/worker_0.log", "logs/worker_1.log",
    "metrics/config.json", "metrics/dataset_summary.json", "metrics/client_class_distribution.csv",
    "metrics/history_pretrain.csv", "metrics/history_pretrain.json",
    "metrics/feature_selection_records.csv", "metrics/encoder_history.csv", "metrics/ppo_history.csv",
    "metrics/candidate_evaluations.csv", "metrics/selected_features.json", "metrics/history_round.csv",
    "metrics/history_client.csv", "metrics/history_local_epoch.csv", "metrics/evaluation_metrics.csv",
    "metrics/evaluation_metrics.json", "metrics/final_round_metrics.csv", "metrics/final_round_metrics.json",
    "metrics/server_evaluation_status.json", "metrics/summary.json",
    "metrics/classification_report.json", "metrics/classification_report.csv",
    "metrics/confusion_matrix.csv", "metrics/confusion_matrix.npy",
    "metrics/personalized_test_metrics.json", "metrics/communication_costs.json",
    "metrics/runtime_breakdown.json", "artifacts/class_distribution.png",
    "artifacts/evaluation_metric_curves.png", "artifacts/loss_curves.png",
    "artifacts/confusion_matrix.png", "artifacts/per_class_f1.png",
    "artifacts/runtime_per_round.png", "artifacts/communication_cumulative.png",
    "artifacts/gpu_utilization.png",
    "artifacts/subset_size_search.png", "artifacts/selected_feature_importance.png",
]
missing = [relative for relative in required if not (RUN_DIR / relative).is_file() or (RUN_DIR / relative).stat().st_size == 0]
assert not missing, f"Missing/empty outputs: {missing}"
assert CONFUSION_MATRIX_DIR.is_dir()
assert len(pd.read_csv(RUN_DIR / "metrics/history_round.csv")) == 10
assert len(pd.read_csv(RUN_DIR / "metrics/history_client.csv")) == 100
assert len(pd.read_csv(RUN_DIR / "metrics/history_local_epoch.csv")) == 100
evaluation = pd.read_csv(RUN_DIR / "metrics/evaluation_metrics.csv")
assert len(evaluation) == 100
assert set(evaluation["round"]) == set(range(1, 11))
assert set(evaluation["model_scope"]) == {"client"}
assert all(name in evaluation.columns for name in CONFIG["common_metrics"])
runtime_module.verify_checkpoint_manifest(RUN_DIR, CONFIG, evaluation.to_dict("records"))
runtime_module.verify_evaluation_metrics(evaluation.to_dict("records"), CONFIG, int(evaluation["test_examples"].iloc[0]))
summary = json.loads((RUN_DIR / "metrics/summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "complete"
print(json.dumps({
    "selected_features": summary["selected_features"],
    "weighted_test_metrics": summary["final_weighted_mean_personalized_test_metrics"],
    "runtime": summary["runtime"],
}, indent=2, ensure_ascii=False))
for artifact in [
    "subset_size_search.png", "selected_feature_importance.png", "evaluation_metric_curves.png",
    "loss_curves.png", "confusion_matrix.png", "per_class_f1.png",
    "runtime_per_round.png", "communication_cumulative.png", "gpu_utilization.png", "class_distribution.png",
]:
    display(Image(filename=str(RUN_DIR / "artifacts" / artifact)))
"""
    return {
        "cells": [
            markdown_cell(title),
            markdown_cell("## 1. Environment gate\n\nAssert chính xác hai GPU NVIDIA T4 trước mọi công việc nặng.\n"),
            code_cell(environment_cell),
            markdown_cell("## 2. Serialized experiment contract\n\nMọi thông số paper và quyết định triển khai nằm trong một `CONFIG`.\n"),
            code_cell(config_cell),
            markdown_cell("## 3. Self-contained client-parallel runtime\n\nRuntime dùng CPU coordinator và hai persistent CUDA worker; consolidated outputs chỉ do coordinator ghi.\n"),
            code_cell(runtime_cell),
            code_cell(manifest_cell),
            markdown_cell("## 4. Execute FedCAPS\n\nĐóng gói manifest rồi chạy entry point như ordinary subprocess để CUDA multiprocessing an toàn trong Kaggle.\n"),
            code_cell(launch_cell),
            markdown_cell("## 5. Acceptance checks and artifacts\n\nKiểm tra file không rỗng, exact history accounting và hiển thị plot để đối chiếu.\n"),
            code_cell(verification_cell),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "gpuCount": 2, "internet": False},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
    compile(runtime_source, str(RUNTIME_PATH), "exec")
    outputs = []
    for scenario in SCENARIOS:
        folder = ROOT / scenario["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        notebook_path = folder / scenario["notebook"]
        notebook_path.write_text(json.dumps(notebook_for(scenario, runtime_source), indent=1, ensure_ascii=False), encoding="utf-8")
        outputs.append(str(notebook_path.relative_to(ROOT)))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
