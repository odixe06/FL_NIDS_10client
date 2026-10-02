#!/usr/bin/env python3
"""Generate the three self-contained Kaggle notebooks requested for FD-IDS."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent


ROOT = Path(__file__).resolve().parents[1]

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

SPECS = {
    "gru": {
        "folder": "gru_10_clients",
        "filename": "fd_ids_gru_10_clients.ipynb",
        "run_name": "fd_ids_ciciot2023_10c_gru",
        "title": "GRU",
        "expected_parameters": 40034,
        "model_hparams": {
            "input_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
        },
    },
    "transformer": {
        "folder": "transformer_10_clients",
        "filename": "fd_ids_transformer_10_clients.ipynb",
        "run_name": "fd_ids_ciciot2023_10c_transformer",
        "title": "Transformer",
        "expected_parameters": 71010,
        "model_hparams": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "pooling": "mean",
        },
    },
    "cnn1d": {
        "folder": "cnn1d_10_clients",
        "filename": "fd_ids_cnn1d_10_clients.ipynb",
        "run_name": "fd_ids_ciciot2023_10c_cnn1d",
        "title": "CNN-1D",
        "expected_parameters": 35874,
        "model_hparams": {
            "channels": [32, 64, 128],
            "kernel_size": 3,
            "pooling": "max_then_adaptive_average",
        },
    },
}


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def guarded_code(source: str, stage: str, on_error: str = "") -> str:
    body = indent(dedent(source).strip(), "    ")
    handler = ""
    if on_error.strip():
        handler = "\n" + indent(dedent(on_error).strip(), "    ")
    return (
        "try:\n"
        f"{body}\n"
        "except Exception:\n"
        f'    logger.error("{stage} failed: %s", traceback.format_exc())'
        f"{handler}\n"
        "    raise"
    )


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INTRO_TEMPLATE = r"""
# FD-IDS — 10-client __TITLE__ trên CICIoT2023 Non-IID

Notebook này tái hiện phương pháp chính của paper **FD-IDS: A Federated
Learning and Knowledge Distillation-Based Intrusion Detection System for
Non-IID IoT Environments** với dữ liệu trong `data_description.md`.

- Mô hình: **__TITLE__** cho cả global teacher và local students.
- 10 client, FedProx, round-wise knowledge distillation.
- 10 communication rounds, 1 local epoch.
- Global batch size 1024, tương đương danh định 512 mẫu trên mỗi GPU T4.
- Toàn bộ dữ liệu client được sử dụng; mỗi client giữ lại 5% theo lớp làm
  validation.
- `global_test_data.csv` chỉ được đánh giá một lần sau khi nạp `best.pt`.
- PyTorch, đúng hai GPU T4, `DistributedDataParallel`/NCCL, CUDA mixed
  precision; mỗi GPU chạy trong một process riêng.

> **Lưu ý về feature selection:** 25 đặc trưng đầu vào đã được chọn upstream
> bằng XGBoost và chuẩn hóa bằng `QuantileTransformer`. Notebook giữ công thức
> Mutual Information của paper để đối chiếu phương pháp nhưng không tuyên bố
> chạy lại MI khi dữ liệu 46 đặc trưng gốc không có trong Kaggle input.
"""


FORMULAS = r"""
## Phương pháp và công thức

Mutual Information trong paper:

$$
I(V_x;V_y)=E(V_x)+E(V_y)-JE(V_x,V_y) \tag{1}
$$

Server tổng hợp có trọng số theo số mẫu local train:

$$
w_G^{t+1}=\sum_{k=1}^{K}\frac{n_k}{n}w_k^{t+1},
\qquad n=\sum_{k=1}^{K}n_k. \tag{2}
$$

FedProx giới hạn độ lệch của local student so với global teacher ở đầu vòng:

$$
L_{\mathrm{proximal}}=
\frac{\mu}{2}\left\|w_k-w_G^t\right\|_2^2. \tag{3}
$$

Teacher cung cấp soft targets ở nhiệt độ $T$:

$$
L_{\mathrm{soft}}=
T^2 KL\left(
\operatorname{softmax}(Z_t/T)
\parallel
\operatorname{softmax}(Z_s/T)
\right). \tag{4}
$$

Nếu chỉ dùng FedAvg và KD:

$$
L_{\mathrm{distill}}=
\lambda L_{\mathrm{hard}}+(1-\lambda)L_{\mathrm{soft}}. \tag{5}
$$

Phương pháp chạy trong notebook là FedProx + round-wise KD:

$$
L_{\mathrm{total}}=
\lambda L_{\mathrm{hard}}+
(1-\lambda)L_{\mathrm{soft}}+
\beta L_{\mathrm{proximal}}, \tag{6}
$$

với $\mu=0.01$, $\lambda=0.5$, $\beta=0.1$ và $T=3$. Global model ở đầu
mỗi vòng được giữ cố định làm teacher trong toàn bộ local update của vòng đó.

Paper mô hình hóa phân bố nhãn client bằng:

$$
\mathbf{p}^{(k)}=(p_1^k,\ldots,p_C^k)\sim\operatorname{Dir}(\theta). \tag{7}
$$

Các file hiện tại đã được chia sẵn cho 10 client với $\alpha=0.2$; notebook
không chia Dirichlet lại.

Các metric:

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}, \tag{8}
$$

$$
\mathrm{Precision}=\frac{TP}{TP+FP}, \tag{9}
$$

$$
\mathrm{Recall}=\frac{TP}{TP+FN}, \tag{10}
$$

$$
\mathrm{F1}=2\frac{\mathrm{Precision}\times\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}, \tag{11}
$$

$$
\mathrm{FPR}=\frac{FP}{FP+TN}, \tag{12}
$$

$$
\mathrm{FNR}=\frac{FN}{FN+TP}. \tag{13}
$$

Notebook báo cáo cả macro/weighted multiclass metrics và binary
`BENIGN`-versus-attack metrics. Mọi tỷ lệ được lưu dưới dạng số thực trong
`[0,1]`.
"""


ENVIRONMENT = r"""
# Kaggle accelerator check: this intentionally fails unless the session is T4 x2.
import platform
import sys

import torch

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

n_gpu = torch.cuda.device_count()
for gpu_index in range(n_gpu):
    print(f"GPU[{gpu_index}]: {torch.cuda.get_device_name(gpu_index)}")

assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle settings."
assert n_gpu == 2, "Set the Kaggle accelerator to GPU T4 x2."
assert all("T4" in torch.cuda.get_device_name(i) for i in range(n_gpu)), (
    "This notebook requires exactly two NVIDIA T4 GPUs."
)

"""


CONFIG_TEMPLATE = r"""
import gc
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

FEATURE_COLUMNS = __FEATURE_COLUMNS__

CONFIG = {
    "input_dir": "/kaggle/input/datasets/odixe0502/data-for-10clients",
    "run_name": "__RUN_NAME__",
    "model_family": "__MODEL_KEY__",
    "model_hparams": __MODEL_HPARAMS__,
    "expected_trainable_parameters": __EXPECTED_PARAMETERS__,
    "num_clients": 10,
    "num_classes": 34,
    "label_column": "Label",
    "feature_columns": FEATURE_COLUMNS,
    "validation_fraction": 0.05,
    "communication_rounds": 10,
    "local_epochs": 1,
    "batch_size": 1024,
    "batch_size_per_gpu": 512,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.001,
    "learning_rate_scaling": "none_adam_preserve_paper_lr",
    "optimizer": "Adam",
    "scheduler": None,
    "mu": 0.01,
    "lambda_hard": 0.5,
    "beta": 0.1,
    "temperature": 3.0,
    "world_size": 2,
    "distributed_backend": "nccl",
    "distributed_launcher": "torchrun",
    "multi_gpu": "DistributedDataParallel",
    "num_workers_per_process": 2,
    "num_workers_total": 4,
    "prefetch_factor": 2,
    "omp_num_threads_per_process": 1,
    "csv_chunk_rows": 262_144,
    "seed": 42,
    "aggregation": "sample_weighted_fedavg",
    "knowledge_distillation": "round_wise",
    "class_imbalance_handling": "none",
    "dataset_partition": "10_clients_prepartitioned_dirichlet_alpha_0.2",
    "feature_selection_upstream": "XGBoost_top_25",
    "scaling_upstream": "QuantileTransformer_train_fit",
}

INPUT_DIR = Path(CONFIG["input_dir"])
OUT = Path("/kaggle/working") / CONFIG["run_name"]
CACHE = Path("/kaggle/temp") / f"{CONFIG['run_name']}_cache"
for subdirectory in ("checkpoints", "logs", "metrics", "artifacts"):
    (OUT / subdirectory).mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

assert CONFIG["batch_size"] % n_gpu == 0
assert CONFIG["batch_size_per_gpu"] == CONFIG["batch_size"] // n_gpu
assert CONFIG["gradient_accumulation_steps"] == 1
assert CONFIG["world_size"] == n_gpu == 2
assert (
    CONFIG["num_workers_total"]
    == CONFIG["world_size"] * CONFIG["num_workers_per_process"]
)

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
logger = logging.getLogger("fd_ids")


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

ENVIRONMENT = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
write_json(OUT / "metrics" / "config.json", CONFIG)
logger.info("Run %s started", CONFIG["run_name"])
logger.info("Environment: %s", json.dumps(ENVIRONMENT, sort_keys=True))
logger.info("Config: %s", json.dumps(json_ready(CONFIG), sort_keys=True))
"""


DATA_PREPARATION = r"""
## Validate the dataset contract and create disk-backed arrays

The CSV files are converted once to NumPy memmaps under `/kaggle/temp`. This
keeps all rows while avoiding simultaneous in-memory pandas DataFrames. The
split is exactly stratified within each client whenever a class has at least
two samples.
"""


DATA_CODE = r"""
pipeline_started = time.perf_counter()
preprocess_started = time.perf_counter()

CLIENT_FILES = [
    INPUT_DIR / f"client_{client_id}_train.csv"
    for client_id in range(1, CONFIG["num_clients"] + 1)
]
TEST_FILE = INPUT_DIR / "global_test_data.csv"
GLOBAL_TRAIN_FILE = INPUT_DIR / "global_train_data.csv"
LABEL_MAPPING_FILE = INPUT_DIR / "label_mapping.csv"
REQUIRED_FILES = CLIENT_FILES + [TEST_FILE, GLOBAL_TRAIN_FILE, LABEL_MAPPING_FILE]

missing_files = [str(path) for path in REQUIRED_FILES if not path.is_file()]
assert not missing_files, f"Missing required input files: {missing_files}"

EXPECTED_COLUMNS = FEATURE_COLUMNS + [CONFIG["label_column"]]
for csv_path in CLIENT_FILES + [TEST_FILE, GLOBAL_TRAIN_FILE]:
    actual_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    assert actual_columns == EXPECTED_COLUMNS, (
        f"Unexpected columns/order in {csv_path.name}.\n"
        f"Expected: {EXPECTED_COLUMNS}\nActual: {actual_columns}"
    )

label_mapping_frame = pd.read_csv(LABEL_MAPPING_FILE)
assert label_mapping_frame.columns.tolist() == ["Encoded_ID", "Label_Name"]
assert sorted(label_mapping_frame["Encoded_ID"].astype(int).tolist()) == list(
    range(CONFIG["num_classes"])
)
LABEL_MAPPING = {
    int(row.Encoded_ID): str(row.Label_Name)
    for row in label_mapping_frame.itertuples(index=False)
}
CLASS_NAMES = [LABEL_MAPPING[class_id] for class_id in range(CONFIG["num_classes"])]
BENIGN_ID = next(
    class_id for class_id, class_name in LABEL_MAPPING.items()
    if class_name.upper() == "BENIGN"
)
logger.info("Validated input schema: %d features, %d classes, BENIGN id=%d",
            len(FEATURE_COLUMNS), len(CLASS_NAMES), BENIGN_ID)


def count_csv_rows(path, block_bytes=64 * 1024 * 1024):
    # Count data rows without parsing the full CSV into memory.
    newline_count = 0
    last_byte = b""
    bytes_seen = 0
    next_log = 4 * 1024**3
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
            bytes_seen += len(block)
            if bytes_seen >= next_log:
                logger.info("Counting %s: %.2f GiB scanned", Path(path).name,
                            bytes_seen / 1024**3)
                next_log += 4 * 1024**3
    logical_lines = newline_count + int(bool(last_byte) and last_byte != b"\n")
    rows = logical_lines - 1
    assert rows > 0, f"No data rows in {path}"
    return rows


def convert_csv_to_memmap(csv_path, cache_prefix, validation_fraction=None, seed=42):
    # Convert one CSV and optionally build an exact per-class train/val split.
    x_path = CACHE / f"{cache_prefix}_features.npy"
    y_path = CACHE / f"{cache_prefix}_labels.npy"
    train_index_path = CACHE / f"{cache_prefix}_train_indices.npy"
    val_index_path = CACHE / f"{cache_prefix}_val_indices.npy"
    metadata_path = CACHE / f"{cache_prefix}_metadata.json"

    source_signature = {
        "path": str(csv_path),
        "size_bytes": csv_path.stat().st_size,
        "mtime_ns": csv_path.stat().st_mtime_ns,
    }
    required_cache = [x_path, y_path, metadata_path]
    if validation_fraction is not None:
        required_cache += [train_index_path, val_index_path]

    if all(path.exists() for path in required_cache):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_signature") == source_signature:
            logger.info("Reusing validated cache for %s", csv_path.name)
            return metadata

    row_count = count_csv_rows(csv_path)
    logger.info("Converting %s: %s rows", csv_path.name, f"{row_count:,}")
    features_map = open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, len(FEATURE_COLUMNS)),
    )
    labels_map = open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(row_count,),
    )

    dtype_map = {column: np.float32 for column in FEATURE_COLUMNS}
    dtype_map[CONFIG["label_column"]] = np.int64
    offset = 0
    class_counts = np.zeros(CONFIG["num_classes"], dtype=np.int64)

    for chunk_number, frame in enumerate(
        pd.read_csv(
            csv_path,
            usecols=EXPECTED_COLUMNS,
            dtype=dtype_map,
            chunksize=CONFIG["csv_chunk_rows"],
        ),
        start=1,
    ):
        feature_values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        label_values = frame[CONFIG["label_column"]].to_numpy(dtype=np.int64, copy=False)
        assert np.isfinite(feature_values).all(), (
            f"NaN or infinite feature value in {csv_path.name}, chunk {chunk_number}"
        )
        assert ((label_values >= 0) & (label_values < CONFIG["num_classes"])).all(), (
            f"Label outside [0, {CONFIG['num_classes'] - 1}] in {csv_path.name}"
        )
        end = offset + len(frame)
        features_map[offset:end] = feature_values
        labels_map[offset:end] = label_values
        class_counts += np.bincount(
            label_values, minlength=CONFIG["num_classes"]
        ).astype(np.int64)
        offset = end
        if chunk_number == 1 or chunk_number % 10 == 0:
            logger.info(
                "Converted %s: %s/%s rows",
                csv_path.name,
                f"{offset:,}",
                f"{row_count:,}",
            )

    assert offset == row_count, f"Row-count mismatch for {csv_path.name}"
    features_map.flush()
    labels_map.flush()
    del features_map, labels_map

    metadata = {
        "source_signature": source_signature,
        "features_path": str(x_path),
        "labels_path": str(y_path),
        "rows": int(row_count),
        "class_counts": class_counts.tolist(),
        "train_indices_path": None,
        "val_indices_path": None,
        "train_rows": int(row_count),
        "val_rows": 0,
        "train_class_counts": class_counts.tolist(),
        "val_class_counts": np.zeros(CONFIG["num_classes"], dtype=np.int64).tolist(),
    }

    if validation_fraction is not None:
        labels_readonly = np.load(y_path, mmap_mode="r")
        validation_mask = np.zeros(row_count, dtype=bool)
        validation_counts = np.zeros(CONFIG["num_classes"], dtype=np.int64)
        rng = np.random.default_rng(seed)
        for class_id in range(CONFIG["num_classes"]):
            class_indices = np.flatnonzero(labels_readonly == class_id)
            if class_indices.size >= 2:
                class_validation_rows = int(round(class_indices.size * validation_fraction))
                class_validation_rows = max(1, class_validation_rows)
                class_validation_rows = min(class_validation_rows, class_indices.size - 1)
                rng.shuffle(class_indices)
                chosen = class_indices[:class_validation_rows]
                validation_mask[chosen] = True
                validation_counts[class_id] = class_validation_rows
            del class_indices

        validation_indices = np.flatnonzero(validation_mask).astype(np.int64)
        train_indices = np.flatnonzero(~validation_mask).astype(np.int64)
        np.save(train_index_path, train_indices, allow_pickle=False)
        np.save(val_index_path, validation_indices, allow_pickle=False)
        train_counts = class_counts - validation_counts
        metadata.update(
            {
                "train_indices_path": str(train_index_path),
                "val_indices_path": str(val_index_path),
                "train_rows": int(train_indices.size),
                "val_rows": int(validation_indices.size),
                "train_class_counts": train_counts.tolist(),
                "val_class_counts": validation_counts.tolist(),
            }
        )
        del labels_readonly, validation_mask, validation_indices, train_indices
        gc.collect()

    write_json(metadata_path, metadata)
    return metadata


client_metadata = []
for client_id, client_file in enumerate(CLIENT_FILES, start=1):
    metadata = convert_csv_to_memmap(
        client_file,
        cache_prefix=f"client_{client_id}",
        validation_fraction=CONFIG["validation_fraction"],
        seed=CONFIG["seed"] + client_id,
    )
    metadata["client_id"] = client_id
    client_metadata.append(metadata)
    logger.info(
        "Client %02d ready: train=%s validation=%s",
        client_id,
        f"{metadata['train_rows']:,}",
        f"{metadata['val_rows']:,}",
    )

test_metadata = convert_csv_to_memmap(
    TEST_FILE,
    cache_prefix="global_test",
    validation_fraction=None,
    seed=CONFIG["seed"],
)

global_train_rows = count_csv_rows(GLOBAL_TRAIN_FILE)
client_total_rows = sum(item["rows"] for item in client_metadata)
assert global_train_rows == client_total_rows, (
    "The sum of client rows does not equal global_train_data.csv: "
    f"{client_total_rows:,} != {global_train_rows:,}"
)

distribution_rows = []
for metadata in client_metadata:
    for class_id, class_name in enumerate(CLASS_NAMES):
        distribution_rows.append(
            {
                "client_id": metadata["client_id"],
                "class_id": class_id,
                "class_name": class_name,
                "total_count": metadata["class_counts"][class_id],
                "train_count": metadata["train_class_counts"][class_id],
                "validation_count": metadata["val_class_counts"][class_id],
            }
        )
distribution_frame = pd.DataFrame(distribution_rows)
distribution_frame.to_csv(
    OUT / "metrics" / "client_class_distribution.csv", index=False
)

DATASET_SUMMARY = {
    "input_dir": str(INPUT_DIR),
    "feature_columns": FEATURE_COLUMNS,
    "num_features": len(FEATURE_COLUMNS),
    "num_classes": CONFIG["num_classes"],
    "label_mapping": LABEL_MAPPING,
    "benign_id": BENIGN_ID,
    "global_train_rows": int(global_train_rows),
    "client_total_rows": int(client_total_rows),
    "local_train_rows": int(sum(item["train_rows"] for item in client_metadata)),
    "validation_rows": int(sum(item["val_rows"] for item in client_metadata)),
    "test_rows": int(test_metadata["rows"]),
    "validation_fraction": CONFIG["validation_fraction"],
    "clients": [
        {
            "client_id": item["client_id"],
            "total_rows": item["rows"],
            "train_rows": item["train_rows"],
            "validation_rows": item["val_rows"],
            "total_class_counts": item["class_counts"],
            "train_class_counts": item["train_class_counts"],
            "validation_class_counts": item["val_class_counts"],
        }
        for item in client_metadata
    ],
    "test_class_counts": test_metadata["class_counts"],
}
write_json(OUT / "metrics" / "dataset_summary.json", DATASET_SUMMARY)
preprocessing_seconds = time.perf_counter() - preprocess_started
logger.info(
    "Dataset ready in %.2f s | train=%s validation=%s test=%s",
    preprocessing_seconds,
    f"{DATASET_SUMMARY['local_train_rows']:,}",
    f"{DATASET_SUMMARY['validation_rows']:,}",
    f"{DATASET_SUMMARY['test_rows']:,}",
)
"""


DATASET_AND_METRICS = r"""
class IndexedMemmapDataset(Dataset):
    # Disk-backed map-style dataset safe to reopen inside DataLoader workers.

    def __init__(self, features_path, labels_path, indices_path=None):
        self.features_path = str(features_path)
        self.labels_path = str(labels_path)
        self.indices_path = str(indices_path) if indices_path is not None else None
        labels = np.load(self.labels_path, mmap_mode="r")
        self.length = int(
            np.load(self.indices_path, mmap_mode="r").shape[0]
            if self.indices_path is not None
            else labels.shape[0]
        )
        self._features = None
        self._labels = None
        self._indices = None

    def __len__(self):
        return self.length

    def _open(self):
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r")
            self._labels = np.load(self.labels_path, mmap_mode="r")
            if self.indices_path is not None:
                self._indices = np.load(self.indices_path, mmap_mode="r")

    def __getitem__(self, item):
        self._open()
        row_index = int(self._indices[item]) if self._indices is not None else int(item)
        features = torch.from_numpy(
            np.asarray(self._features[row_index], dtype=np.float32).copy()
        )
        target = torch.tensor(int(self._labels[row_index]), dtype=torch.long)
        return features, target

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_labels"] = None
        state["_indices"] = None
        return state


def dataset_from_metadata(metadata, split):
    if split == "train":
        indices_path = metadata["train_indices_path"]
    elif split == "validation":
        indices_path = metadata["val_indices_path"]
    elif split == "test":
        indices_path = None
    else:
        raise ValueError(f"Unsupported split: {split}")
    return IndexedMemmapDataset(
        metadata["features_path"],
        metadata["labels_path"],
        indices_path,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": CONFIG["batch_size"],
        "shuffle": shuffle,
        "num_workers": CONFIG["num_workers_per_process"],
        "pin_memory": True,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if CONFIG["num_workers_per_process"] > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    false_negative = support - true_positive
    false_positive = predicted - true_positive
    total = float(matrix.sum())
    true_negative = total - true_positive - false_negative - false_positive

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    fpr = safe_divide(false_positive, false_positive + true_negative)
    fnr = safe_divide(false_negative, false_negative + true_positive)
    weights = safe_divide(support, support.sum())

    attack_ids = [class_id for class_id in range(CONFIG["num_classes"])
                  if class_id != BENIGN_ID]
    binary_tp = float(matrix[np.ix_(attack_ids, attack_ids)].sum())
    binary_fn = float(matrix[attack_ids, BENIGN_ID].sum())
    binary_fp = float(matrix[BENIGN_ID, attack_ids].sum())
    binary_tn = float(matrix[BENIGN_ID, BENIGN_ID])
    binary_precision = float(safe_divide(binary_tp, binary_tp + binary_fp))
    binary_recall = float(safe_divide(binary_tp, binary_tp + binary_fn))
    binary_f1 = float(
        safe_divide(
            2.0 * binary_precision * binary_recall,
            binary_precision + binary_recall,
        )
    )

    return {
        "accuracy": float(safe_divide(true_positive.sum(), total)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "multiclass_macro_fpr": float(fpr.mean()),
        "multiclass_macro_fnr": float(fnr.mean()),
        "binary_attack_accuracy": float(
            safe_divide(binary_tp + binary_tn, binary_tp + binary_tn + binary_fp + binary_fn)
        ),
        "binary_attack_precision": binary_precision,
        "binary_attack_recall": binary_recall,
        "binary_attack_f1": binary_f1,
        "binary_attack_fpr": float(safe_divide(binary_fp, binary_fp + binary_tn)),
        "binary_attack_fnr": float(safe_divide(binary_fn, binary_fn + binary_tp)),
    }


def update_confusion(matrix, targets, predictions):
    encoded = targets.astype(np.int64) * CONFIG["num_classes"] + predictions.astype(np.int64)
    matrix += np.bincount(
        encoded,
        minlength=CONFIG["num_classes"] ** 2,
    ).reshape(CONFIG["num_classes"], CONFIG["num_classes"])


def classification_report_from_confusion(matrix):
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = safe_divide(true_positive, predicted)
    recall = safe_divide(true_positive, support)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    weights = safe_divide(support, support.sum())
    report = {}
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = {
            "class_id": class_id,
            "class_name": class_name,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1_score": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        report[str(class_id)] = values
        rows.append(values)
    aggregate_rows = [
        {
            "class_id": "macro_avg",
            "class_name": "macro_avg",
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1_score": float(f1.mean()),
            "support": int(support.sum()),
        },
        {
            "class_id": "weighted_avg",
            "class_name": "weighted_avg",
            "precision": float((precision * weights).sum()),
            "recall": float((recall * weights).sum()),
            "f1_score": float((f1 * weights).sum()),
            "support": int(support.sum()),
        },
    ]
    report["accuracy"] = float(safe_divide(true_positive.sum(), support.sum()))
    report["macro_avg"] = aggregate_rows[0]
    report["weighted_avg"] = aggregate_rows[1]
    rows.extend(aggregate_rows)
    return report, pd.DataFrame(rows)


@torch.no_grad()
def evaluate_dataset(model, dataset, seed):
    model.eval()
    loader = make_loader(dataset, shuffle=False, seed=seed)
    matrix = np.zeros(
        (CONFIG["num_classes"], CONFIG["num_classes"]), dtype=np.int64
    )
    loss_sum = 0.0
    examples = 0
    started = time.perf_counter()
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=1)
        batch_examples = targets.size(0)
        loss_sum += float(loss.item()) * batch_examples
        examples += batch_examples
        update_confusion(
            matrix,
            targets.detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
        )
    elapsed = time.perf_counter() - started
    assert examples == len(dataset)
    return {
        "loss": loss_sum / max(examples, 1),
        "examples": examples,
        "metrics": metrics_from_confusion(matrix),
        "confusion_matrix": matrix,
        "seconds": elapsed,
    }
"""


MODEL_CODE = {
    "gru": r"""
class GRUClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, features):
        sequence = features.unsqueeze(-1)
        encoded, _ = self.gru(sequence)
        return self.classifier(encoded[:, -1, :])


def build_model():
    return GRUClassifier(len(FEATURE_COLUMNS), CONFIG["num_classes"])
""",
    "transformer": r"""
class TransformerClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        d_model = 64
        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_features, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, features):
        tokens = self.input_projection(features.unsqueeze(-1))
        tokens = tokens + self.position_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(self.output_norm(pooled))


def build_model():
    return TransformerClassifier(len(FEATURE_COLUMNS), CONFIG["num_classes"])
""",
    "cnn1d": r"""
class CNN1DClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, features):
        encoded = self.features(features.unsqueeze(1)).squeeze(-1)
        return self.classifier(encoded)


def build_model():
    return CNN1DClassifier(len(FEATURE_COLUMNS), CONFIG["num_classes"])
""",
}


MODEL_SETUP = r"""
def unwrap(model):
    return model.module if hasattr(model, "module") else model


global_model = build_model().to(device)

trainable_parameters = sum(
    parameter.numel()
    for parameter in unwrap(global_model).parameters()
    if parameter.requires_grad
)
total_parameters = sum(
    parameter.numel() for parameter in unwrap(global_model).parameters()
)
model_state_bytes = sum(
    tensor.numel() * tensor.element_size()
    for tensor in unwrap(global_model).state_dict().values()
)
assert trainable_parameters == CONFIG["expected_trainable_parameters"], (
    f"Parameter-count regression: expected {CONFIG['expected_trainable_parameters']:,}, "
    f"found {trainable_parameters:,}"
)
MODEL_METADATA = {
    "model_family": CONFIG["model_family"],
    "model_hparams": CONFIG["model_hparams"],
    "trainable_parameters": int(trainable_parameters),
    "total_parameters": int(total_parameters),
    "model_state_bytes": int(model_state_bytes),
    "model_state_mib": float(model_state_bytes / 1024**2),
}
logger.info("Model metadata: %s", json.dumps(MODEL_METADATA, sort_keys=True))
"""


TRAINING_HELPERS = r"""
def new_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def proximal_loss(student_core, teacher_core):
    squared_distance = torch.zeros((), device=device)
    for student_parameter, teacher_parameter in zip(
        student_core.parameters(), teacher_core.parameters()
    ):
        squared_distance = squared_distance + torch.sum(
            (student_parameter - teacher_parameter.detach()) ** 2
        )
    return 0.5 * CONFIG["mu"] * squared_distance


def train_one_client(global_teacher, metadata, round_number):
    client_id = metadata["client_id"]
    dataset = dataset_from_metadata(metadata, "train")
    loader = make_loader(
        dataset,
        shuffle=True,
        seed=CONFIG["seed"] + round_number * 10_000 + client_id,
    )
    student = build_model().to(device)
    unwrap(student).load_state_dict(unwrap(global_teacher).state_dict())
    optimizer = torch.optim.Adam(student.parameters(), lr=CONFIG["learning_rate"])
    scaler = new_grad_scaler()
    global_teacher.eval()

    local_epoch_rows = []
    overall_sums = {"hard": 0.0, "soft": 0.0, "proximal": 0.0, "total": 0.0}
    overall_examples = 0
    client_started = time.perf_counter()

    for local_epoch in range(1, CONFIG["local_epochs"] + 1):
        student.train()
        epoch_sums = {"hard": 0.0, "soft": 0.0, "proximal": 0.0, "total": 0.0}
        epoch_examples = 0
        batch_count = 0
        epoch_started = time.perf_counter()
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    teacher_logits = global_teacher(features)

            with torch.amp.autocast("cuda"):
                student_logits = student(features)
                hard_loss = F.cross_entropy(student_logits, targets)
                soft_loss = (
                    F.kl_div(
                        F.log_softmax(
                            student_logits / CONFIG["temperature"], dim=1
                        ),
                        F.softmax(
                            teacher_logits / CONFIG["temperature"], dim=1
                        ),
                        reduction="batchmean",
                    )
                    * CONFIG["temperature"] ** 2
                )
            prox_loss = proximal_loss(unwrap(student), unwrap(global_teacher))
            total_loss = (
                CONFIG["lambda_hard"] * hard_loss
                + (1.0 - CONFIG["lambda_hard"]) * soft_loss
                + CONFIG["beta"] * prox_loss
            )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_examples = targets.size(0)
            values = {
                "hard": float(hard_loss.item()),
                "soft": float(soft_loss.item()),
                "proximal": float(prox_loss.item()),
                "total": float(total_loss.item()),
            }
            for key, value in values.items():
                epoch_sums[key] += value * batch_examples
                overall_sums[key] += value * batch_examples
            epoch_examples += batch_examples
            overall_examples += batch_examples
            batch_count += 1

        epoch_seconds = time.perf_counter() - epoch_started
        epoch_row = {
            "round": round_number,
            "client_id": client_id,
            "local_epoch": local_epoch,
            "train_examples": epoch_examples,
            "batches": batch_count,
            "hard_loss": epoch_sums["hard"] / max(epoch_examples, 1),
            "soft_loss": epoch_sums["soft"] / max(epoch_examples, 1),
            "proximal_loss": epoch_sums["proximal"] / max(epoch_examples, 1),
            "total_loss": epoch_sums["total"] / max(epoch_examples, 1),
            "epoch_seconds": epoch_seconds,
        }
        local_epoch_rows.append(epoch_row)
        logger.info(
            "Round %02d/%02d | client %02d/%02d | epoch %d/%d | "
            "hard=%.6f soft=%.6f prox=%.6f total=%.6f | %.2f s",
            round_number,
            CONFIG["communication_rounds"],
            client_id,
            CONFIG["num_clients"],
            local_epoch,
            CONFIG["local_epochs"],
            epoch_row["hard_loss"],
            epoch_row["soft_loss"],
            epoch_row["proximal_loss"],
            epoch_row["total_loss"],
            epoch_seconds,
        )

    local_validation_dataset = dataset_from_metadata(metadata, "validation")
    local_validation_result = evaluate_dataset(
        student,
        local_validation_dataset,
        seed=CONFIG["seed"] + round_number * 10_000 + client_id + 1_000_000,
    )
    client_seconds = time.perf_counter() - client_started
    local_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in unwrap(student).state_dict().items()
    }
    client_stats = {
        "round": round_number,
        "client_id": client_id,
        "train_rows": metadata["train_rows"],
        "validation_rows": metadata["val_rows"],
        "train_examples_processed": overall_examples,
        "hard_loss": overall_sums["hard"] / max(overall_examples, 1),
        "soft_loss": overall_sums["soft"] / max(overall_examples, 1),
        "proximal_loss": overall_sums["proximal"] / max(overall_examples, 1),
        "total_loss": overall_sums["total"] / max(overall_examples, 1),
        "local_train_seconds": (
            client_seconds - local_validation_result["seconds"]
        ),
        "local_validation_loss": local_validation_result["loss"],
        "local_validation_seconds": local_validation_result["seconds"],
    }
    for metric_name, metric_value in local_validation_result["metrics"].items():
        client_stats[f"local_validation_{metric_name}"] = metric_value
    logger.info(
        "Round %02d/%02d | client %02d local validation | "
        "accuracy=%.6f macro_f1=%.6f | %.2f s",
        round_number,
        CONFIG["communication_rounds"],
        client_id,
        local_validation_result["metrics"]["accuracy"],
        local_validation_result["metrics"]["macro_f1"],
        local_validation_result["seconds"],
    )
    del (
        student,
        optimizer,
        scaler,
        loader,
        dataset,
        local_validation_dataset,
        local_validation_result,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return local_state, client_stats, local_epoch_rows


def add_weighted_state(aggregate, local_state, weight, first_client):
    for name, tensor in local_state.items():
        if torch.is_floating_point(tensor):
            contribution = tensor.to(torch.float32).mul(weight)
            if first_client:
                aggregate[name] = contribution
            else:
                aggregate[name].add_(contribution)
        elif first_client:
            aggregate[name] = tensor.clone()
        else:
            aggregate[name] = torch.maximum(aggregate[name], tensor)


def finalize_aggregate(aggregate, reference_state):
    finalized = {}
    for name, reference_tensor in reference_state.items():
        finalized[name] = aggregate[name].to(dtype=reference_tensor.dtype)
    return finalized


def checkpoint_payload(round_number, validation_metrics):
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in unwrap(global_model).state_dict().items()
        },
        "round": int(round_number),
        "config": json_ready(CONFIG),
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": LABEL_MAPPING,
        "validation_metrics": json_ready(validation_metrics),
        "model_metadata": MODEL_METADATA,
    }


def persist_histories(round_history, client_history, local_epoch_history):
    pd.DataFrame(round_history).to_csv(
        OUT / "metrics" / "history_round.csv", index=False
    )
    pd.DataFrame(client_history).to_csv(
        OUT / "metrics" / "history_client.csv", index=False
    )
    pd.DataFrame(local_epoch_history).to_csv(
        OUT / "metrics" / "history_local_epoch.csv", index=False
    )
    write_json(OUT / "metrics" / "history_round.json", round_history)
    write_json(OUT / "metrics" / "history_client.json", client_history)
    write_json(OUT / "metrics" / "history_local_epoch.json", local_epoch_history)
"""


TRAINING_LOOP = r"""
round_history = []
client_history = []
local_epoch_history = []
best_validation_macro_f1 = -math.inf
best_round = None
best_validation_metrics = None
communication_bytes_per_round = (
    2 * CONFIG["num_clients"] * MODEL_METADATA["model_state_bytes"]
)
cumulative_communication_bytes = 0
total_train_rows = sum(item["train_rows"] for item in client_metadata)
training_started = time.perf_counter()

try:
    for round_number in range(1, CONFIG["communication_rounds"] + 1):
        round_started = time.perf_counter()
        for gpu_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gpu_index)
        aggregate = {}
        pending_client_rows = []

        for client_position, metadata in enumerate(client_metadata):
            local_state, client_stats, epoch_rows = train_one_client(
                global_model, metadata, round_number
            )
            client_weight = metadata["train_rows"] / total_train_rows
            add_weighted_state(
                aggregate,
                local_state,
                weight=client_weight,
                first_client=(client_position == 0),
            )
            pending_client_rows.append(client_stats)
            local_epoch_history.extend(epoch_rows)
            del local_state

        reference_state = unwrap(global_model).state_dict()
        aggregated_state = finalize_aggregate(aggregate, reference_state)
        unwrap(global_model).load_state_dict(aggregated_state, strict=True)
        del aggregate, aggregated_state
        gc.collect()

        global_validation_matrix = np.zeros(
            (CONFIG["num_classes"], CONFIG["num_classes"]), dtype=np.int64
        )
        global_validation_loss_sum = 0.0
        global_validation_examples = 0
        for metadata, client_row in zip(client_metadata, pending_client_rows):
            validation_dataset = dataset_from_metadata(metadata, "validation")
            validation_result = evaluate_dataset(
                global_model,
                validation_dataset,
                seed=CONFIG["seed"] + round_number * 10_000 + metadata["client_id"],
            )
            global_validation_matrix += validation_result["confusion_matrix"]
            global_validation_loss_sum += (
                validation_result["loss"] * validation_result["examples"]
            )
            global_validation_examples += validation_result["examples"]
            client_row["global_validation_loss"] = validation_result["loss"]
            client_row["global_validation_seconds"] = validation_result["seconds"]
            for metric_name, metric_value in validation_result["metrics"].items():
                client_row[f"global_validation_{metric_name}"] = metric_value
            client_history.append(client_row)
            del validation_dataset, validation_result

        validation_metrics = metrics_from_confusion(global_validation_matrix)
        validation_loss = global_validation_loss_sum / max(
            global_validation_examples, 1
        )
        weighted_train = {}
        for loss_name in ("hard_loss", "soft_loss", "proximal_loss", "total_loss"):
            weighted_train[loss_name] = sum(
                row[loss_name] * row["train_rows"] for row in pending_client_rows
            ) / total_train_rows

        round_seconds = time.perf_counter() - round_started
        cumulative_training_seconds = time.perf_counter() - training_started
        cumulative_communication_bytes += communication_bytes_per_round
        gpu_peak_bytes = [
            int(torch.cuda.max_memory_allocated(gpu_index))
            for gpu_index in range(torch.cuda.device_count())
        ]
        round_row = {
            "round": round_number,
            "train_hard_loss": weighted_train["hard_loss"],
            "train_soft_loss": weighted_train["soft_loss"],
            "train_proximal_loss": weighted_train["proximal_loss"],
            "train_total_loss": weighted_train["total_loss"],
            "best_local_client_accuracy": max(
                row["local_validation_accuracy"] for row in pending_client_rows
            ),
            "worst_local_client_accuracy": min(
                row["local_validation_accuracy"] for row in pending_client_rows
            ),
            "mean_local_client_accuracy": float(
                np.mean(
                    [
                        row["local_validation_accuracy"]
                        for row in pending_client_rows
                    ]
                )
            ),
            "best_local_client_macro_f1": max(
                row["local_validation_macro_f1"] for row in pending_client_rows
            ),
            "worst_local_client_macro_f1": min(
                row["local_validation_macro_f1"] for row in pending_client_rows
            ),
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "round_seconds": round_seconds,
            "cumulative_training_seconds": cumulative_training_seconds,
            "communication_bytes_round": communication_bytes_per_round,
            "communication_mib_round": communication_bytes_per_round / 1024**2,
            "communication_bytes_cumulative": cumulative_communication_bytes,
            "communication_mib_cumulative": cumulative_communication_bytes / 1024**2,
            "cuda_gpu0_peak_memory_bytes": gpu_peak_bytes[0],
            "cuda_gpu0_peak_memory_mib": gpu_peak_bytes[0] / 1024**2,
            "cuda_gpu1_peak_memory_bytes": gpu_peak_bytes[1],
            "cuda_gpu1_peak_memory_mib": gpu_peak_bytes[1] / 1024**2,
            "cuda_peak_memory_max_device_bytes": max(gpu_peak_bytes),
            "cuda_peak_memory_max_device_mib": max(gpu_peak_bytes) / 1024**2,
            "cuda_peak_memory_sum_devices_bytes": sum(gpu_peak_bytes),
            "cuda_peak_memory_sum_devices_mib": sum(gpu_peak_bytes) / 1024**2,
        }
        round_history.append(round_row)

        torch.save(
            checkpoint_payload(round_number, validation_metrics),
            OUT / "checkpoints" / "last.pt",
        )
        if validation_metrics["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = validation_metrics["macro_f1"]
            best_round = round_number
            best_validation_metrics = {
                "loss": validation_loss,
                **validation_metrics,
            }
            torch.save(
                checkpoint_payload(round_number, best_validation_metrics),
                OUT / "checkpoints" / "best.pt",
            )
            logger.info(
                "Saved best.pt at round %d with validation macro-F1 %.6f",
                round_number,
                best_validation_macro_f1,
            )

        persist_histories(round_history, client_history, local_epoch_history)
        logger.info(
            "Round %02d/%02d complete | val_loss=%.6f val_acc=%.6f "
            "val_macro_f1=%.6f | %.2f s | comm=%.3f MiB cumulative",
            round_number,
            CONFIG["communication_rounds"],
            validation_loss,
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            round_seconds,
            cumulative_communication_bytes / 1024**2,
        )

    training_and_validation_seconds = time.perf_counter() - training_started
    logger.info(
        "Federated training completed in %.2f s | best round=%d macro-F1=%.6f",
        training_and_validation_seconds,
        best_round,
        best_validation_macro_f1,
    )
except Exception:
    training_and_validation_seconds = time.perf_counter() - training_started
    logger.error("Federated training failed:\n%s", traceback.format_exc())
    write_json(
        OUT / "metrics" / "summary.json",
        {
            "run_name": CONFIG["run_name"],
            "model_family": CONFIG["model_family"],
            "status": "failed",
            "config": CONFIG,
            "model_metadata": MODEL_METADATA,
            "dataset": DATASET_SUMMARY,
            "completed_rounds": len(round_history),
            "runtime_seconds": {
                "preprocessing_seconds": preprocessing_seconds,
                "training_and_validation_seconds": training_and_validation_seconds,
            },
            "traceback": traceback.format_exc(),
        },
    )
    raise
"""


FINAL_EVALUATION = r"""
## Reload the best checkpoint and evaluate the untouched global test set

The test confusion matrix is accumulated batch-by-batch; individual predictions
are intentionally not persisted because the full test set contains millions of
rows.
"""


FINAL_CODE = r"""
try:
    try:
        best_checkpoint = torch.load(
            OUT / "checkpoints" / "best.pt",
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        best_checkpoint = torch.load(
            OUT / "checkpoints" / "best.pt",
            map_location=device,
        )
    unwrap(global_model).load_state_dict(
        best_checkpoint["model_state_dict"], strict=True
    )
    logger.info("Reloaded best.pt from round %d", best_checkpoint["round"])

    test_dataset = dataset_from_metadata(test_metadata, "test")
    final_test_started = time.perf_counter()
    test_result = evaluate_dataset(
        global_model,
        test_dataset,
        seed=CONFIG["seed"] + 999_999,
    )
    final_test_seconds = time.perf_counter() - final_test_started
    test_matrix = test_result["confusion_matrix"]
    test_metrics = {
        "loss": test_result["loss"],
        **test_result["metrics"],
    }

    np.save(
        OUT / "metrics" / "confusion_matrix.npy",
        test_matrix,
        allow_pickle=False,
    )
    confusion_frame = pd.DataFrame(
        test_matrix,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    )
    confusion_frame.to_csv(OUT / "metrics" / "confusion_matrix.csv")

    classification_report, classification_report_frame = (
        classification_report_from_confusion(test_matrix)
    )
    write_json(
        OUT / "metrics" / "classification_report.json",
        classification_report,
    )
    classification_report_frame.to_csv(
        OUT / "metrics" / "classification_report.csv", index=False
    )

    communication_rows = []
    for round_number in range(1, CONFIG["communication_rounds"] + 1):
        communication_rows.append(
            {
                "round": round_number,
                "model_state_bytes": MODEL_METADATA["model_state_bytes"],
                "participating_clients": CONFIG["num_clients"],
                "download_bytes": (
                    CONFIG["num_clients"] * MODEL_METADATA["model_state_bytes"]
                ),
                "upload_bytes": (
                    CONFIG["num_clients"] * MODEL_METADATA["model_state_bytes"]
                ),
                "total_bytes_round": communication_bytes_per_round,
                "total_mib_round": communication_bytes_per_round / 1024**2,
                "cumulative_bytes": communication_bytes_per_round * round_number,
                "cumulative_mib": (
                    communication_bytes_per_round * round_number / 1024**2
                ),
            }
        )
    communication_frame = pd.DataFrame(communication_rows)
    communication_frame.to_csv(
        OUT / "metrics" / "communication_costs.csv", index=False
    )
    COMMUNICATION_SUMMARY = {
        "estimation_scope": (
            "raw model state only; server-to-client download plus client-to-server "
            "upload; excludes protocol, serialization, optimizer, retry and compression"
        ),
        "model_state_bytes": MODEL_METADATA["model_state_bytes"],
        "model_state_mib": MODEL_METADATA["model_state_mib"],
        "participating_clients_per_round": CONFIG["num_clients"],
        "rounds": CONFIG["communication_rounds"],
        "download_bytes_per_round": (
            CONFIG["num_clients"] * MODEL_METADATA["model_state_bytes"]
        ),
        "upload_bytes_per_round": (
            CONFIG["num_clients"] * MODEL_METADATA["model_state_bytes"]
        ),
        "total_bytes_per_round": communication_bytes_per_round,
        "total_mib_per_round": communication_bytes_per_round / 1024**2,
        "total_bytes_all_rounds": (
            communication_bytes_per_round * CONFIG["communication_rounds"]
        ),
        "total_mib_all_rounds": (
            communication_bytes_per_round
            * CONFIG["communication_rounds"]
            / 1024**2
        ),
    }
    write_json(
        OUT / "metrics" / "communication_costs.json",
        COMMUNICATION_SUMMARY,
    )

    total_notebook_pipeline_seconds = time.perf_counter() - pipeline_started
    RUNTIME_BREAKDOWN = {
        "preprocessing_seconds": preprocessing_seconds,
        "training_and_validation_seconds": training_and_validation_seconds,
        "final_test_seconds": final_test_seconds,
        "total_notebook_pipeline_seconds": total_notebook_pipeline_seconds,
        "sum_local_client_training_seconds": float(
            sum(row["local_train_seconds"] for row in client_history)
        ),
        "sum_local_client_validation_seconds": float(
            sum(row["local_validation_seconds"] for row in client_history)
        ),
        "sum_global_client_validation_seconds": float(
            sum(row["global_validation_seconds"] for row in client_history)
        ),
        "per_round_seconds": [
            {"round": row["round"], "seconds": row["round_seconds"]}
            for row in round_history
        ],
        "per_client_round_training_seconds": [
            {
                "round": row["round"],
                "client_id": row["client_id"],
                "seconds": row["local_train_seconds"],
            }
            for row in client_history
        ],
    }
    write_json(
        OUT / "metrics" / "runtime_breakdown.json",
        RUNTIME_BREAKDOWN,
    )

    OUTPUT_FILES = {
        "best_checkpoint": "checkpoints/best.pt",
        "last_checkpoint": "checkpoints/last.pt",
        "run_log": "logs/run.log",
        "config": "metrics/config.json",
        "dataset_summary": "metrics/dataset_summary.json",
        "client_class_distribution": "metrics/client_class_distribution.csv",
        "round_history_csv": "metrics/history_round.csv",
        "round_history_json": "metrics/history_round.json",
        "client_history_csv": "metrics/history_client.csv",
        "client_history_json": "metrics/history_client.json",
        "local_epoch_history_csv": "metrics/history_local_epoch.csv",
        "local_epoch_history_json": "metrics/history_local_epoch.json",
        "classification_report_json": "metrics/classification_report.json",
        "classification_report_csv": "metrics/classification_report.csv",
        "confusion_matrix_csv": "metrics/confusion_matrix.csv",
        "confusion_matrix_npy": "metrics/confusion_matrix.npy",
        "communication_costs_json": "metrics/communication_costs.json",
        "communication_costs_csv": "metrics/communication_costs.csv",
        "runtime_breakdown": "metrics/runtime_breakdown.json",
        "class_distribution_plot": "artifacts/class_distribution.png",
        "accuracy_f1_plot": "artifacts/accuracy_f1_curves.png",
        "loss_plot": "artifacts/loss_curves.png",
        "confusion_matrix_plot": "artifacts/confusion_matrix.png",
        "per_class_f1_plot": "artifacts/per_class_f1.png",
        "runtime_plot": "artifacts/runtime_per_round.png",
        "communication_plot": "artifacts/communication_cumulative.png",
    }
    SUMMARY = {
        "run_name": CONFIG["run_name"],
        "model_family": CONFIG["model_family"],
        "status": "evaluation_completed",
        "config": CONFIG,
        "environment": ENVIRONMENT,
        "model_metadata": MODEL_METADATA,
        "dataset": DATASET_SUMMARY,
        "best_round": best_round,
        "best_validation_metrics": best_validation_metrics,
        "final_test_metrics": test_metrics,
        "runtime_seconds": RUNTIME_BREAKDOWN,
        "communication": COMMUNICATION_SUMMARY,
        "metric_units": {
            "rates": "fraction_in_[0,1]",
            "time": "seconds",
            "communication_binary": "MiB (1 MiB = 1,048,576 bytes)",
            "communication_exact": "bytes",
        },
        "outputs": OUTPUT_FILES,
    }
    write_json(OUT / "metrics" / "summary.json", SUMMARY)
    logger.info("Final test metrics: %s", json.dumps(test_metrics, sort_keys=True))
    logger.info("Final evaluation completed in %.2f s", final_test_seconds)
except Exception:
    logger.error("Final evaluation failed:\n%s", traceback.format_exc())
    raise
"""


PLOTS = r"""
# Diagnostic plots: save PNG and render inline.
plotting_started = time.perf_counter()
round_frame = pd.DataFrame(round_history)

class_count_matrix = (
    distribution_frame.pivot(
        index="client_id", columns="class_name", values="total_count"
    )
    .reindex(index=range(1, CONFIG["num_clients"] + 1), columns=CLASS_NAMES)
    .fillna(0)
    .to_numpy()
)
plt.figure(figsize=(18, 5.5))
plt.imshow(np.log10(class_count_matrix + 1), aspect="auto", cmap="magma")
plt.colorbar(label="log10(count + 1)")
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90, fontsize=7)
plt.yticks(range(CONFIG["num_clients"]), [f"Client {i}" for i in range(1, 11)])
plt.xlabel("Class")
plt.ylabel("Client")
plt.title("Client class distribution (all pre-split rows)")
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "class_distribution.png", dpi=160)
plt.show()

plt.figure(figsize=(8, 4.8))
plt.plot(
    round_frame["round"],
    round_frame["validation_accuracy"],
    marker="o",
    label="Validation accuracy",
)
plt.plot(
    round_frame["round"],
    round_frame["validation_macro_f1"],
    marker="s",
    label="Validation macro-F1",
)
plt.plot(
    round_frame["round"],
    round_frame["validation_weighted_f1"],
    marker="^",
    label="Validation weighted-F1",
)
plt.xlabel("Communication round")
plt.ylabel("Score [0,1]")
plt.title(f"{CONFIG['model_family'].upper()} validation accuracy/F1")
plt.xticks(round_frame["round"])
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "accuracy_f1_curves.png", dpi=160)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(round_frame["round"], round_frame["train_hard_loss"], label="Hard loss")
plt.plot(round_frame["round"], round_frame["train_soft_loss"], label="Soft KD loss")
plt.plot(
    round_frame["round"],
    round_frame["train_proximal_loss"],
    label="FedProx loss",
)
plt.plot(round_frame["round"], round_frame["train_total_loss"], label="Total loss")
plt.plot(
    round_frame["round"],
    round_frame["validation_loss"],
    linestyle="--",
    label="Validation CE loss",
)
plt.xlabel("Communication round")
plt.ylabel("Mean loss")
plt.title("Training loss components and validation loss")
plt.xticks(round_frame["round"])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "loss_curves.png", dpi=160)
plt.show()

normalized_matrix = safe_divide(
    test_matrix,
    test_matrix.sum(axis=1, keepdims=True),
)
plt.figure(figsize=(16, 14))
plt.imshow(normalized_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(fraction=0.046, pad=0.04, label="Row-normalized rate")
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90, fontsize=7)
plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=7)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Global test confusion matrix (row-normalized)")
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "confusion_matrix.png", dpi=180)
plt.show()

class_report_plot = classification_report_frame[
    classification_report_frame["class_id"].apply(
        lambda value: isinstance(value, (int, np.integer))
    )
].copy()
plt.figure(figsize=(12, 6))
plt.bar(
    class_report_plot["class_name"],
    class_report_plot["f1_score"],
    color="steelblue",
)
plt.xticks(rotation=90, fontsize=7)
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Test F1 [0,1]")
plt.title("Per-class F1 on global test")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "per_class_f1.png", dpi=160)
plt.show()

plt.figure(figsize=(8, 4.8))
plt.bar(round_frame["round"], round_frame["round_seconds"], color="slateblue")
plt.xlabel("Communication round")
plt.ylabel("Wall time (seconds)")
plt.title("Runtime per federated round")
plt.xticks(round_frame["round"])
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "runtime_per_round.png", dpi=160)
plt.show()

plt.figure(figsize=(8, 4.8))
plt.plot(
    round_frame["round"],
    round_frame["communication_mib_cumulative"],
    marker="o",
    color="darkgreen",
)
plt.xlabel("Communication round")
plt.ylabel("Estimated cumulative communication (MiB)")
plt.title("Raw model-state communication estimate")
plt.xticks(round_frame["round"])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "communication_cumulative.png", dpi=160)
plt.show()

plotting_seconds = time.perf_counter() - plotting_started
RUNTIME_BREAKDOWN["plotting_seconds"] = plotting_seconds
RUNTIME_BREAKDOWN["total_notebook_pipeline_seconds"] = (
    time.perf_counter() - pipeline_started
)
SUMMARY["status"] = "completed"
SUMMARY["runtime_seconds"] = RUNTIME_BREAKDOWN
write_json(OUT / "metrics" / "runtime_breakdown.json", RUNTIME_BREAKDOWN)
write_json(OUT / "metrics" / "summary.json", SUMMARY)
logger.info("All required artifacts were saved under %s", OUT)
for output_path in sorted(OUT.rglob("*")):
    if output_path.is_file():
        logger.info(
            "OUTPUT %s | %s bytes",
            output_path.relative_to(OUT),
            f"{output_path.stat().st_size:,}",
        )
"""


DDP_LAUNCH_MARKDOWN = r"""
## Huấn luyện federated bằng DistributedDataParallel

Cell dưới ghi một entry point Python vào cache và khởi chạy chính xác hai
process bằng `torchrun`: rank 0 dùng GPU 0, rank 1 dùng GPU 1. Mỗi process nhận
batch 512, nên effective global batch là 1024. Gradient được đồng bộ bằng NCCL
trong mỗi optimizer step; server FedAvg logic chỉ chạy ở rank 0 rồi broadcast
global state sang rank 1.

`DistributedSampler` có thể đệm tối đa một dòng cho client có số mẫu train lẻ
để hai rank luôn có cùng số optimizer step. Không dòng nào bị loại.
Validation và test dùng sampler riêng không padding, vì vậy mỗi mẫu được tính
đúng một lần.
"""


DDP_LAUNCH_CODE_TEMPLATE = r"""
import subprocess

DDP_RUNTIME_SOURCE = __DDP_RUNTIME_SOURCE__
DDP_MANIFEST = {
    "config": CONFIG,
    "client_metadata": client_metadata,
    "test_metadata": test_metadata,
    "dataset_summary": DATASET_SUMMARY,
    "label_mapping": LABEL_MAPPING,
    "benign_id": BENIGN_ID,
    "out_dir": str(OUT),
    "preprocessing_seconds": preprocessing_seconds,
    "environment": ENVIRONMENT,
}
DDP_SCRIPT_PATH = CACHE / "fd_ids_ddp_runtime.py"
DDP_MANIFEST_PATH = CACHE / "fd_ids_ddp_manifest.json"
DDP_SCRIPT_PATH.write_text(DDP_RUNTIME_SOURCE, encoding="utf-8")
write_json(DDP_MANIFEST_PATH, DDP_MANIFEST)

launch_command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nnodes=1",
    "--max-restarts=0",
    f"--nproc-per-node={CONFIG['world_size']}",
    str(DDP_SCRIPT_PATH),
]
launch_environment = os.environ.copy()
launch_environment.update(
    {
        "FD_IDS_MANIFEST": str(DDP_MANIFEST_PATH),
        "OMP_NUM_THREADS": str(CONFIG["omp_num_threads_per_process"]),
        "MKL_NUM_THREADS": str(CONFIG["omp_num_threads_per_process"]),
        "NCCL_DEBUG": "WARN",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
)
logger.info("Launching DDP: %s", " ".join(launch_command))
for handler in logger.handlers:
    handler.flush()
logging.shutdown()

torchrun_started = time.perf_counter()
try:
    subprocess.run(
        launch_command,
        check=True,
        env=launch_environment,
    )
except Exception:
    failure_text = "DDP launcher failed:\n" + traceback.format_exc()
    with LOG_FILE.open("a", encoding="utf-8") as log_handle:
        log_handle.write(failure_text + "\n")
    print(failure_text, file=sys.stderr)
    raise
torchrun_wall_seconds = time.perf_counter() - torchrun_started

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
logger = logging.getLogger("fd_ids")

required_outputs = [
    OUT / "checkpoints" / "best.pt",
    OUT / "checkpoints" / "last.pt",
    OUT / "logs" / "run.log",
    OUT / "logs" / "rank_1.log",
    OUT / "metrics" / "config.json",
    OUT / "metrics" / "dataset_summary.json",
    OUT / "metrics" / "client_class_distribution.csv",
    OUT / "metrics" / "history_round.csv",
    OUT / "metrics" / "history_round.json",
    OUT / "metrics" / "history_client.csv",
    OUT / "metrics" / "history_client.json",
    OUT / "metrics" / "history_local_epoch.csv",
    OUT / "metrics" / "history_local_epoch.json",
    OUT / "metrics" / "summary.json",
    OUT / "metrics" / "classification_report.json",
    OUT / "metrics" / "classification_report.csv",
    OUT / "metrics" / "confusion_matrix.csv",
    OUT / "metrics" / "confusion_matrix.npy",
    OUT / "metrics" / "runtime_breakdown.json",
    OUT / "metrics" / "communication_costs.json",
    OUT / "metrics" / "communication_costs.csv",
    OUT / "artifacts" / "class_distribution.png",
    OUT / "artifacts" / "accuracy_f1_curves.png",
    OUT / "artifacts" / "loss_curves.png",
    OUT / "artifacts" / "confusion_matrix.png",
    OUT / "artifacts" / "per_class_f1.png",
    OUT / "artifacts" / "runtime_per_round.png",
    OUT / "artifacts" / "communication_cumulative.png",
]
missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
assert not missing_outputs, f"DDP completed but outputs are missing: {missing_outputs}"
empty_outputs = [str(path) for path in required_outputs if path.stat().st_size == 0]
assert not empty_outputs, f"DDP completed but outputs are empty: {empty_outputs}"

round_frame = pd.read_csv(OUT / "metrics" / "history_round.csv")
client_frame = pd.read_csv(OUT / "metrics" / "history_client.csv")
local_epoch_frame = pd.read_csv(OUT / "metrics" / "history_local_epoch.csv")
assert len(round_frame) == CONFIG["communication_rounds"]
assert len(client_frame) == (
    CONFIG["communication_rounds"] * CONFIG["num_clients"]
)
assert len(local_epoch_frame) == (
    CONFIG["communication_rounds"]
    * CONFIG["num_clients"]
    * CONFIG["local_epochs"]
)
assert np.load(
    OUT / "metrics" / "confusion_matrix.npy",
    allow_pickle=False,
).shape == (CONFIG["num_classes"], CONFIG["num_classes"])

runtime_breakdown = json.loads(
    (OUT / "metrics" / "runtime_breakdown.json").read_text(encoding="utf-8")
)
runtime_breakdown["torchrun_wall_seconds"] = torchrun_wall_seconds
runtime_breakdown["total_notebook_pipeline_seconds"] = (
    time.perf_counter() - pipeline_started
)
write_json(OUT / "metrics" / "runtime_breakdown.json", runtime_breakdown)
summary = json.loads(
    (OUT / "metrics" / "summary.json").read_text(encoding="utf-8")
)
assert summary["status"] == "completed"
assert summary["config"]["communication_rounds"] == 10
assert summary["config"]["local_epochs"] == 1
assert summary["config"]["batch_size"] == 1024
summary["runtime_seconds"] = runtime_breakdown
write_json(OUT / "metrics" / "summary.json", summary)
logger.info(
    "DDP subprocess completed in %.2f s; all required outputs validated under %s",
    torchrun_wall_seconds,
    OUT,
)
"""


INLINE_ARTIFACTS = r"""
from IPython.display import Image, display

artifact_names = [
    "class_distribution.png",
    "accuracy_f1_curves.png",
    "loss_curves.png",
    "confusion_matrix.png",
    "per_class_f1.png",
    "runtime_per_round.png",
    "communication_cumulative.png",
]
for artifact_name in artifact_names:
    artifact_path = OUT / "artifacts" / artifact_name
    print(artifact_name)
    display(Image(filename=str(artifact_path)))

print("Summary:", OUT / "metrics" / "summary.json")
print("Best checkpoint:", OUT / "checkpoints" / "best.pt")
print("Last checkpoint:", OUT / "checkpoints" / "last.pt")
"""


def build_cells(model_key: str, spec: dict) -> list[dict]:
    intro = (
        INTRO_TEMPLATE.replace("__TITLE__", spec["title"])
    )
    config = (
        CONFIG_TEMPLATE
        .replace("__FEATURE_COLUMNS__", repr(FEATURE_COLUMNS))
        .replace("__RUN_NAME__", spec["run_name"])
        .replace("__MODEL_KEY__", model_key)
        .replace("__MODEL_HPARAMS__", repr(spec["model_hparams"]))
        .replace("__EXPECTED_PARAMETERS__", str(spec["expected_parameters"]))
    )
    ddp_runtime_source = (
        ROOT / "tools" / "fd_ids_ddp_runtime.py"
    ).read_text(encoding="utf-8")
    ddp_launch_code = DDP_LAUNCH_CODE_TEMPLATE.replace(
        "__DDP_RUNTIME_SOURCE__",
        repr(ddp_runtime_source),
    )
    return [
        markdown_cell(intro),
        markdown_cell(FORMULAS),
        code_cell(ENVIRONMENT),
        code_cell(config),
        markdown_cell(DATA_PREPARATION),
        code_cell(guarded_code(DATA_CODE, "Dataset preparation")),
        markdown_cell(
            f"""
            ## Kiến trúc {spec['title']}

            Model nhận `[batch, 25]`, trả logits `[batch, 34]`. Số tham số
            trainable được kiểm tra trong DDP runtime để ngăn thay đổi kiến trúc
            ngoài ý muốn. Chi tiết layer và tensor shape nằm trong
            `ARCHITECTURE_AND_OUTPUT_SPEC.md`.
            """
        ),
        markdown_cell(DDP_LAUNCH_MARKDOWN),
        code_cell(ddp_launch_code),
        markdown_cell(
            """
            ## Diagnostic plots

            Bảy biểu đồ đã được process rank 0 lưu thành PNG. Cell sau tải lại
            và hiển thị chúng inline trong notebook. Các bảng số liệu nguồn nằm
            trong `metrics/`.
            """
        ),
        code_cell(INLINE_ARTIFACTS),
        markdown_cell(
            """
            ## Hoàn tất

            Khi cell cuối chạy thành công, `metrics/summary.json` là entry point
            chuẩn cho report. Notebook không lưu toàn bộ dự đoán theo từng dòng
            nhằm tránh output quá lớn; confusion matrix và classification
            report giữ đủ thống kê đánh giá đã yêu cầu.
            """
        ),
    ]


def main() -> None:
    for model_key, spec in SPECS.items():
        destination_dir = ROOT / spec["folder"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / spec["filename"]
        payload = notebook(build_cells(model_key, spec))
        destination.write_text(
            json.dumps(payload, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
