#!/usr/bin/env python3
"""Generate three self-contained Kaggle T4x2 client-parallel notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PARALLEL_RUNTIME_SOURCE = (
    ROOT / "scripts" / "client_parallel_runtime.py"
).read_text(encoding="utf-8")

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

SCENARIOS = [
    {
        "folder": "scenario_1_10_clients_gru",
        "notebook": "fd_ids_ciciot2023_10c_gru.ipynb",
        "run_name": "fd_ids_ciciot2023_10c_gru",
        "scenario_name": "10 clients GRU",
        "client_architectures": ["gru"] * 10,
    },
    {
        "folder": "scenario_2_10_clients_transformer",
        "notebook": "fd_ids_ciciot2023_10c_transformer.ipynb",
        "run_name": "fd_ids_ciciot2023_10c_transformer",
        "scenario_name": "10 clients Transformer",
        "client_architectures": ["transformer"] * 10,
    },
    {
        "folder": "scenario_3_5_gru_5_transformer",
        "notebook": "fd_ids_ciciot2023_10c_mixed_5gru_5transformer.ipynb",
        "run_name": "fd_ids_ciciot2023_10c_mixed_5gru_5transformer",
        "scenario_name": "5 clients GRU + 5 clients Transformer",
        "client_architectures": ["gru"] * 5 + ["transformer"] * 5,
    },
]


PREPROCESS_SOURCE = r'''
def validate_header(csv_path):
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    expected = FEATURE_COLUMNS + ["Label"]
    if columns != expected:
        raise ValueError(
            f"Unexpected columns in {csv_path.name}.\n"
            f"Expected: {expected}\nObserved: {columns}"
        )


def scan_labels(csv_path):
    validate_header(csv_path)
    counts = np.zeros(CONFIG["num_classes"], dtype=np.int64)
    row_count = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=["Label"],
        dtype={"Label": np.int64},
        chunksize=CONFIG["csv_chunk_rows"],
    ):
        labels = chunk["Label"].to_numpy(dtype=np.int64, copy=False)
        if labels.size and (labels.min() < 0 or labels.max() >= CONFIG["num_classes"]):
            raise ValueError(f"Label outside [0, 33] in {csv_path}")
        counts += np.bincount(labels, minlength=CONFIG["num_classes"])
        row_count += labels.size
    return int(row_count), counts


def csv_to_memmaps(csv_path, x_path, y_path, row_count):
    x_map = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, len(FEATURE_COLUMNS)),
    )
    y_map = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(row_count,),
    )
    offset = 0
    for chunk in pd.read_csv(
        csv_path,
        dtype={**{name: np.float32 for name in FEATURE_COLUMNS}, "Label": np.int64},
        chunksize=CONFIG["csv_chunk_rows"],
    ):
        features = chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        labels = chunk["Label"].to_numpy(dtype=np.int64, copy=False)
        if not np.isfinite(features).all():
            raise ValueError(f"NaN or infinity found in {csv_path}")
        end = offset + len(chunk)
        x_map[offset:end] = features
        y_map[offset:end] = labels
        offset = end
    if offset != row_count:
        raise RuntimeError(f"Row count changed while reading {csv_path}")
    x_map.flush()
    y_map.flush()
    del x_map, y_map


def make_stratified_indices(y_path, row_count, class_counts, client_id):
    labels = np.load(y_path, mmap_mode="r")
    val_mask_path = CACHE / f"client_{client_id}_val_mask.npy"
    val_mask = np.lib.format.open_memmap(
        val_mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(row_count,),
    )
    val_mask[:] = 0
    val_counts = np.zeros(CONFIG["num_classes"], dtype=np.int64)
    rng = np.random.default_rng(CONFIG["seed"] + 10_000 + client_id)
    for class_id, class_count in enumerate(class_counts.tolist()):
        if class_count <= 1:
            continue
        requested = int(round(class_count * CONFIG["validation_ratio"]))
        val_count = min(class_count - 1, max(1, requested))
        positions = np.flatnonzero(labels == class_id)
        if len(positions) != class_count:
            raise RuntimeError(f"Class count mismatch for client {client_id}")
        chosen = rng.choice(positions, size=val_count, replace=False, shuffle=False)
        val_mask[chosen] = 1
        val_counts[class_id] = val_count
        del positions, chosen
    val_mask.flush()

    train_count = int(row_count - val_counts.sum())
    val_count = int(val_counts.sum())
    train_index_path = CACHE / f"client_{client_id}_train_indices.npy"
    val_index_path = CACHE / f"client_{client_id}_val_indices.npy"
    train_indices = np.lib.format.open_memmap(
        train_index_path,
        mode="w+",
        dtype=np.int64,
        shape=(train_count,),
    )
    val_indices = np.lib.format.open_memmap(
        val_index_path,
        mode="w+",
        dtype=np.int64,
        shape=(val_count,),
    )
    train_indices[:] = np.flatnonzero(val_mask == 0)
    val_indices[:] = np.flatnonzero(val_mask == 1)
    train_indices.flush()
    val_indices.flush()
    del labels, val_mask, train_indices, val_indices
    val_mask_path.unlink()
    return train_index_path, val_index_path, class_counts - val_counts, val_counts


pipeline_started = time.perf_counter()
preprocessing_started = time.perf_counter()
for required_path in CLIENT_FILES + [GLOBAL_TEST_FILE, GLOBAL_TRAIN_FILE, LABEL_FILE]:
    if not required_path.is_file():
        raise FileNotFoundError(f"Required Kaggle input is missing: {required_path}")

label_frame = pd.read_csv(LABEL_FILE)
if label_frame.columns.tolist() != ["Encoded_ID", "Label_Name"]:
    raise ValueError("label_mapping.csv must contain Encoded_ID,Label_Name")
label_frame = label_frame.sort_values("Encoded_ID").reset_index(drop=True)
expected_ids = list(range(CONFIG["num_classes"]))
if label_frame["Encoded_ID"].astype(int).tolist() != expected_ids:
    raise ValueError("Encoded_ID must be exactly 0..33")
LABEL_MAPPING = {
    int(row.Encoded_ID): str(row.Label_Name)
    for row in label_frame.itertuples(index=False)
}
BENIGN_CLASS_ID = next(
    class_id
    for class_id, name in LABEL_MAPPING.items()
    if name.upper() in {"BENIGN", "BENIGNTRAFFIC"}
)

client_scans = {}
for client_id, csv_path in enumerate(CLIENT_FILES, start=1):
    row_count, class_counts = scan_labels(csv_path)
    client_scans[client_id] = {
        "row_count": row_count,
        "class_counts": class_counts,
    }
    logger.info("Scanned client %d: %s rows", client_id, f"{row_count:,}")

global_train_rows, _ = scan_labels(GLOBAL_TRAIN_FILE)
client_row_sum = sum(item["row_count"] for item in client_scans.values())
if global_train_rows != client_row_sum:
    raise RuntimeError(
        f"global_train_data.csv has {global_train_rows:,} rows, "
        f"but clients sum to {client_row_sum:,}"
    )
test_rows, test_class_counts = scan_labels(GLOBAL_TEST_FILE)
logger.info("Scanned global test: %s rows", f"{test_rows:,}")

client_manifest = {}
distribution_rows = []
for client_id, csv_path in enumerate(CLIENT_FILES, start=1):
    row_count = client_scans[client_id]["row_count"]
    class_counts = client_scans[client_id]["class_counts"]
    x_path = CACHE / f"client_{client_id}_features.npy"
    y_path = CACHE / f"client_{client_id}_labels.npy"
    csv_to_memmaps(csv_path, x_path, y_path, row_count)
    train_index_path, val_index_path, train_counts, val_counts = (
        make_stratified_indices(
            y_path,
            row_count,
            class_counts,
            client_id,
        )
    )
    client_manifest[str(client_id)] = {
        "client_id": client_id,
        "architecture": CONFIG["client_architectures"][client_id - 1],
        "features_path": str(x_path),
        "labels_path": str(y_path),
        "train_indices_path": str(train_index_path),
        "val_indices_path": str(val_index_path),
        "original_examples": row_count,
        "train_examples": int(train_counts.sum()),
        "validation_examples": int(val_counts.sum()),
        "original_class_counts": class_counts.tolist(),
        "train_class_counts": train_counts.tolist(),
        "validation_class_counts": val_counts.tolist(),
    }
    for class_id in range(CONFIG["num_classes"]):
        distribution_rows.append(
            {
                "client_id": client_id,
                "architecture": CONFIG["client_architectures"][client_id - 1],
                "class_id": class_id,
                "class_name": LABEL_MAPPING[class_id],
                "original_examples": int(class_counts[class_id]),
                "train_examples": int(train_counts[class_id]),
                "validation_examples": int(val_counts[class_id]),
            }
        )
    logger.info(
        "Cached client %d: train=%s validation=%s",
        client_id,
        f"{int(train_counts.sum()):,}",
        f"{int(val_counts.sum()):,}",
    )

test_x_path = CACHE / "global_test_features.npy"
test_y_path = CACHE / "global_test_labels.npy"
csv_to_memmaps(GLOBAL_TEST_FILE, test_x_path, test_y_path, test_rows)
preprocessing_seconds = time.perf_counter() - preprocessing_started

confirmed_data_manifest = {
    "input_dir": str(INPUT_DIR),
    "clients": client_manifest,
    "test": {
        "features_path": str(test_x_path),
        "labels_path": str(test_y_path),
        "examples": test_rows,
        "class_counts": test_class_counts.tolist(),
    },
    "global_train_rows": global_train_rows,
    "label_mapping": LABEL_MAPPING,
    "benign_class_id": BENIGN_CLASS_ID,
}
dataset_summary = {
    "input_dir": str(INPUT_DIR),
    "feature_columns": FEATURE_COLUMNS,
    "num_features": len(FEATURE_COLUMNS),
    "num_classes": CONFIG["num_classes"],
    "global_train_rows": global_train_rows,
    "client_train_rows_after_validation": sum(
        item["train_examples"] for item in client_manifest.values()
    ),
    "client_validation_rows": sum(
        item["validation_examples"] for item in client_manifest.values()
    ),
    "global_test_rows": test_rows,
    "clients": client_manifest,
    "test_class_counts": test_class_counts.tolist(),
}
(OUT / "metrics" / "config.json").write_text(
    json.dumps(CONFIG, indent=2, sort_keys=True),
    encoding="utf-8",
)
(OUT / "metrics" / "dataset_summary.json").write_text(
    json.dumps(dataset_summary, indent=2),
    encoding="utf-8",
)
pd.DataFrame(distribution_rows).to_csv(
    OUT / "metrics" / "client_class_distribution.csv",
    index=False,
)
logger.info("Preprocessing completed in %.2f seconds", preprocessing_seconds)
'''


LAUNCH_SOURCE = r'''
CLIENT_PARALLEL_SCRIPT_PATH = CACHE / "client_parallel_training.py"
MANIFEST_PATH = CACHE / "training_manifest.json"
CLIENT_PARALLEL_SCRIPT_PATH.write_text(
    CLIENT_PARALLEL_RUNTIME_SOURCE,
    encoding="utf-8",
)
MANIFEST_PATH.write_text(
    json.dumps(
        {
            "config": CONFIG,
            "out_dir": str(OUT),
            "cache_dir": str(CACHE),
            "confirmed_data": confirmed_data_manifest,
            "preprocessing_seconds": preprocessing_seconds,
            "pipeline_started_perf_counter": pipeline_started,
        },
        indent=2,
    ),
    encoding="utf-8",
)

command = [sys.executable, str(CLIENT_PARALLEL_SCRIPT_PATH)]
launch_environment = os.environ.copy()
launch_environment.update(
    {
        "TRAINING_MANIFEST": str(MANIFEST_PATH),
        "OMP_NUM_THREADS": str(CONFIG["omp_num_threads_per_process"]),
        "MKL_NUM_THREADS": str(CONFIG["omp_num_threads_per_process"]),
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
)
logger.info("Launching two persistent client-parallel GPU workers: %s", command)
logging.shutdown()
launch_started = time.perf_counter()
try:
    subprocess.run(command, check=True, env=launch_environment)
except Exception:
    failure = "Client-parallel launcher failed:\n" + traceback.format_exc()
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(failure + "\n")
    print(failure, file=sys.stderr)
    raise
subprocess_wall_seconds = time.perf_counter() - launch_started
'''


VERIFY_SOURCE = r'''
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("training")
runtime_path = OUT / "metrics" / "runtime_breakdown.json"
summary_path = OUT / "metrics" / "summary.json"
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
runtime["subprocess_wall_seconds"] = subprocess_wall_seconds
runtime["total_notebook_pipeline_seconds"] = time.perf_counter() - pipeline_started
runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["runtime_breakdown"] = runtime
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

required_relative_paths = summary["output_paths"]
required_outputs = [OUT / relative for relative in required_relative_paths]
missing = [str(path) for path in required_outputs if not path.is_file()]
empty = [
    str(path)
    for path in required_outputs
    if path.is_file() and path.stat().st_size == 0
]
assert not missing, f"Missing outputs: {missing}"
assert not empty, f"Empty outputs: {empty}"
assert len(pd.read_csv(OUT / "metrics" / "history_round.csv")) == CONFIG[
    "communication_rounds"
]
assert len(pd.read_csv(OUT / "metrics" / "history_client.csv")) == (
    CONFIG["communication_rounds"] * CONFIG["num_clients"]
)
history_round_frame = pd.read_csv(OUT / "metrics" / "history_round.csv")
history_client_frame = pd.read_csv(OUT / "metrics" / "history_client.csv")
history_local_frame = pd.read_csv(OUT / "metrics" / "history_local_epoch.csv")
expected_local_rows = (
    CONFIG["communication_rounds"]
    * CONFIG["num_clients"]
    * CONFIG["local_epochs"]
    + CONFIG["num_clients"]
    * CONFIG["final_personalized_finetune_epochs"]
)
assert len(history_local_frame) == expected_local_rows
assert (history_local_frame["sampler_padding_rows"] == 0).all()
assert set(history_local_frame["gpu_id"].astype(int)) == {0, 1}
assert set(history_local_frame["concurrent_streams"].astype(int)) <= {1, 2}
for gpu_id in range(2):
    for memory_name in ("peak_allocated", "peak_reserved"):
        assert f"gpu_{gpu_id}_{memory_name}_bytes" in history_round_frame
        assert f"gpu_{gpu_id}_{memory_name}_mib" in history_round_frame

required_checkpoint_keys = {
    "model_state_dict",
    "personalized_model_state_dicts",
    "initial_model_state_dicts_by_family",
    "initialization_hashes",
    "round",
    "config",
    "feature_columns",
    "label_mapping",
    "validation_metrics",
    "model_metadata",
}
best_checkpoint = torch.load(
    OUT / "checkpoints" / "best.pt",
    map_location="cpu",
    weights_only=False,
)
last_checkpoint = torch.load(
    OUT / "checkpoints" / "last.pt",
    map_location="cpu",
    weights_only=False,
)
assert required_checkpoint_keys <= set(best_checkpoint)
assert required_checkpoint_keys <= set(last_checkpoint)
assert "personalized_model_state_dicts_before_finetune" in best_checkpoint
assert "personalized_model_state_dicts_before_finetune" not in last_checkpoint
assert len(best_checkpoint["personalized_model_state_dicts"]) == 10
assert not any(
    key.startswith("module.")
    for key in best_checkpoint["model_state_dict"]
)
assert (
    best_checkpoint["initialization_hashes"]
    == last_checkpoint["initialization_hashes"]
    == summary["initialization_hashes"]
)
for family, metadata in summary["model_metadata"].items():
    assert (
        metadata["initialization_sha256"]
        == summary["initialization_hashes"][family]
    )
assert summary["execution_mode"] == CONFIG["execution_mode"] == "client_parallel"
assert CONFIG["per_client_batch_size"] == 1024
logger.info("Verified %d required non-empty outputs", len(required_outputs))
logger.info("Initialization hashes: %s", summary["initialization_hashes"])
logger.info(
    "Final proxy accuracy=%.6f macro-F1=%.6f",
    summary["final_test_metrics"]["accuracy"],
    summary["final_test_metrics"]["macro_f1"],
)

from IPython.display import Image, display

for artifact_name in (
    "class_distribution.png",
    "accuracy_f1_curves.png",
    "loss_curves.png",
    "confusion_matrix.png",
    "per_class_f1.png",
    "runtime_per_round.png",
    "communication_cumulative.png",
):
    display(Image(filename=str(OUT / "artifacts" / artifact_name)))
'''


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def build_config_source(scenario: dict) -> str:
    config = {
        "input_dir": "/kaggle/input/datasets/odixe0502/data-for-10clients",
        "run_name": scenario["run_name"],
        "scenario_name": scenario["scenario_name"],
        "scenario_model_family": (
            "gru"
            if len(set(scenario["client_architectures"])) == 1
            and scenario["client_architectures"][0] == "gru"
            else "transformer"
            if len(set(scenario["client_architectures"])) == 1
            else "mixed_gru_transformer"
        ),
        "client_architectures": scenario["client_architectures"],
        "proxy_model_family": "cnn1d",
        "num_clients": 10,
        "num_features": 25,
        "num_classes": 34,
        "feature_columns": FEATURE_COLUMNS,
        "execution_mode": "client_parallel",
        "gpu_worker_count": 2,
        "communication_rounds": 10,
        "local_epochs": 1,
        "final_personalized_finetune_epochs": 1,
        "per_client_batch_size": 1024,
        "gpu_cache_fraction": 0.7,
        "gpu_cache_chunk_rows": 200000,
        "stream_candidates": [1, 2],
        "stream_min_speedup": 1.03,
        "benchmark_warmup_steps": 3,
        "benchmark_timed_steps": 8,
        "stream_benchmark_warmup_steps": 2,
        "stream_benchmark_timed_steps": 6,
        "gpu_monitor_interval_seconds": 1.0,
        "worker_health_poll_seconds": 10.0,
        "worker_join_timeout_seconds": 30.0,
        "omp_num_threads_per_process": 1,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "scheduler": None,
        "label_loss": "cross_entropy",
        "distillation_discrepancy": "mse_softmax",
        "adaptive_epsilon": 1e-8,
        "validation_ratio": 0.05,
        "class_weighting": None,
        "checkpoint_criterion": "validation_macro_f1",
        "seed": 42,
        "initialization_seed": 42,
        "csv_chunk_rows": 200000,
    }
    config_literal = json.dumps(config, indent=4, ensure_ascii=False).replace(
        ": null",
        ": None",
    )
    return f'''
import json
import logging
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {config_literal}
assert CONFIG["gpu_worker_count"] == n_gpu == 2
assert CONFIG["execution_mode"] == "client_parallel"
assert CONFIG["per_client_batch_size"] == 1024
assert CONFIG["client_architectures"] == {scenario["client_architectures"]!r}

FEATURE_COLUMNS = CONFIG["feature_columns"]
INPUT_DIR = Path(CONFIG["input_dir"])
CLIENT_FILES = [
    INPUT_DIR / f"client_{{client_id}}_train.csv"
    for client_id in range(1, CONFIG["num_clients"] + 1)
]
GLOBAL_TEST_FILE = INPUT_DIR / "global_test_data.csv"
GLOBAL_TRAIN_FILE = INPUT_DIR / "global_train_data.csv"
LABEL_FILE = INPUT_DIR / "label_mapping.csv"
OUT = Path("/kaggle/working") / CONFIG["run_name"]
CACHE = Path("/kaggle/temp") / f"{{CONFIG['run_name']}}_cache"
for directory in ("checkpoints", "logs", "metrics", "artifacts"):
    (OUT / directory).mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "logs" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("training")
logger.info("Run %s started", CONFIG["run_name"])
logger.info("Config: %s", json.dumps(CONFIG, sort_keys=True))
'''


def build_notebook(scenario: dict) -> dict:
    intro = f'''
# {scenario["scenario_name"]}: Lightweight Federated IDS

Notebook này tái hiện phần **adaptive federated mutual learning** của bài báo
“Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring”
trên CICIoT2023. Hai process tồn tại suốt phiên, mỗi process sở hữu một T4 và
huấn luyện một nhóm client độc lập; coordinator chỉ tổng hợp proxy sau mỗi round.

## Những điều chỉnh ngắn so với paper gốc

- NILM hồi quy trên REFIT/REDD được thay bằng phân loại 34 lớp trên dữ liệu
  CICIoT2023 đã chọn 25 đặc trưng và chuẩn hóa sẵn.
- Personalized model của client dùng cấu hình `{scenario["scenario_name"]}`;
  mỗi client đồng thời giữ một proxy CNN-1D `[32, 64, 128]`, kernel 3.
- Không chạy MNAS vì kiến trúc personalized model đã được cố định để so sánh.
- Label loss L2 của paper được thay bằng cross-entropy; sai khác distillation là
  MSE giữa hai phân phối softmax. Chỉ proxy được FedAvg có trọng số.
- Tất cả model cùng họ dùng initial state sinh độc lập từ seed 42; initial state
  và SHA-256 được nhúng trong checkpoint để đối chiếu giữa notebook.
- Client được chia tĩnh theo benchmark workload để cân bằng hai GPU; dữ liệu của
  client được cache trên GPU sở hữu với ngân sách 70% VRAM và LRU xác định khi
  không đủ chỗ. Notebook tự benchmark 1/2 CUDA stream và chỉ dùng 2 khi nhanh hơn.
'''
    formulas = r'''
## Công thức được thực thi

Với `n_k` là số mẫu train sau khi tách validation, proxy được tổng hợp:

$$w_G^{t+1}=\sum_{k=1}^{K}\frac{n_k}{\sum_j n_j}w_{r,k}^{t+1}.$$

Hai label loss:

$$\ell_s=\mathrm{CE}(y,z_s),\qquad \ell_r=\mathrm{CE}(y,z_r).$$

Adaptive mutual distillation, là ánh xạ phân loại của (17)–(19):

$$d=\mathrm{MSE}(\mathrm{softmax}(z_s),\mathrm{softmax}(z_r)),$$

$$\ell_d=\frac{d}{\mathrm{stopgrad}(\ell_s+\ell_r)+10^{-8}},$$

$$L_s=\ell_s+\ell_d,\qquad L_r=\ell_r+\ell_d.$$

Một backward chung trên `ℓs + ℓr + ℓd` truyền đúng một lần gradient
distillation vào mỗi model. Công thức MI/NAS của paper chỉ mang tính mô tả vì
input đã được chọn 25 đặc trưng bằng XGBoost và kiến trúc không còn được search.
'''
    environment = r'''
import platform
import sys

import torch

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
for gpu_index in range(n_gpu):
    properties = torch.cuda.get_device_properties(gpu_index)
    print(
        f"GPU[{gpu_index}]: {properties.name}; "
        f"VRAM={properties.total_memory / 1024**3:.2f} GiB; "
        f"compute={properties.major}.{properties.minor}"
    )
assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle settings."
assert n_gpu == 2, "Set the Kaggle accelerator to GPU T4 x2."
assert all("T4" in torch.cuda.get_device_name(i) for i in range(n_gpu))
'''
    runtime_assignment = (
        "CLIENT_PARALLEL_RUNTIME_SOURCE = "
        + repr(CLIENT_PARALLEL_RUNTIME_SOURCE.strip() + "\n")
    )
    notebook = {
        "cells": [
            markdown_cell(intro),
            code_cell(environment),
            markdown_cell(formulas),
            code_cell(build_config_source(scenario)),
            markdown_cell(
                "## Kiểm tra dữ liệu và tạo cache memmap\n\n"
                "Cell này kiểm tra schema, tách validation theo lớp và không "
                "đưa `global_test_data.csv` vào chọn checkpoint."
            ),
            code_cell(PREPROCESS_SOURCE),
            markdown_cell(
                "## Runtime client-parallel\n\n"
                "Runtime dùng `spawn`, hai worker GPU bền vững và coordinator "
                "CPU; không khởi tạo process group."
            ),
            code_cell(runtime_assignment),
            markdown_cell(
                "## Khởi chạy runtime\n\n"
                "Một subprocess Python thông thường tạo đúng hai worker, gắn "
                "cố định worker 0/1 vào `cuda:0`/`cuda:1`."
            ),
            code_cell(LAUNCH_SOURCE),
            markdown_cell("## Xác minh output và hiển thị biểu đồ"),
            code_cell(VERIFY_SOURCE),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [
                    {
                        "datasetId": "odixe0502/data-for-10clients",
                        "sourceType": "datasetVersion",
                    }
                ],
                "isGpuEnabled": True,
                "isInternetEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main() -> None:
    for scenario in SCENARIOS:
        output_dir = ROOT / scenario["folder"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / scenario["notebook"]
        notebook = build_notebook(scenario)
        output_path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
