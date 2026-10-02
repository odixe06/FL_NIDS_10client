from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TEMPLATE_PATH = ROOT / "notebook_runtime_template.py"

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
        "folder": "10_clients_gru",
        "model_family": "gru",
        "run_name": "fd_ids_ciciot2023_10c_gru",
        "expected_parameter_count": 40034,
        "title": "10 client GRU",
    },
    {
        "folder": "10_clients_transformer",
        "model_family": "transformer",
        "run_name": "fd_ids_ciciot2023_10c_transformer",
        "expected_parameter_count": 71010,
        "title": "10 client Transformer",
    },
    {
        "folder": "10_clients_cnn1d",
        "model_family": "cnn1d",
        "run_name": "fd_ids_ciciot2023_10c_cnn1d",
        "expected_parameter_count": 35874,
        "title": "10 client CNN-1D",
    },
]


def source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(source),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def config_for(scenario: dict) -> dict:
    run_name = scenario["run_name"]
    return {
        "run_name": run_name,
        "model_family": scenario["model_family"],
        "execution_mode": "client_parallel",
        "input_dir": "/kaggle/input/datasets/odixe0502/data-for-10clients",
        "output_dir": f"/kaggle/working/{run_name}",
        "temp_dir": f"/kaggle/temp/{run_name}_runtime",
        "cache_dir": f"/kaggle/temp/{run_name}_cache",
        "manifest_path": f"/kaggle/temp/{run_name}_manifest.json",
        "num_clients": 10,
        "num_features": 25,
        "num_classes": 34,
        "benign_label_id": 1,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": "Label",
        "communication_rounds": 10,
        "local_epochs": 1,
        "server_pretrain_epochs": 1,
        "per_client_batch_size": 1024,
        "evaluation_batch_size": 8192,
        "pretrain_chunk_rows": 262144,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "scheduler": None,
        "gradient_accumulation_steps": 1,
        "distillation": "forward_kl_historical_teacher_to_student",
        "distillation_lambda": 1.0,
        "distillation_temperature": 1.0,
        "server_aggregation": "unweighted_mean_selected_clients",
        "selection_rule": "local_accuracy_strictly_below_all_client_mean",
        "validation_fraction": 0.05,
        "seed": 42,
        "initialization_seed": 42,
        "gpu_cache_fraction": 0.7,
        "stream_candidates": [1, 2],
        "stream_benchmark_steps": 3,
        "worker_count": 2,
        "worker_timeout_seconds": 21600,
        "mixed_precision": "cuda_amp",
        "expected_parameter_count": scenario["expected_parameter_count"],
        "checkpoint_criterion": "global_validation_macro_f1",
        "expected_history_rows": {
            "pretrain": 1,
            "round": 10,
            "client": 100,
            "local_epoch": 100,
        },
    }


def introduction_markdown(scenario: dict) -> str:
    return f"""# PerFed-SKD trên CICIoT2023 — {scenario["title"]}

Notebook này tái xây dựng phương pháp **Personalized Federated Learning for
Heterogeneous Edge Device: Self-Knowledge Distillation Approach** trên dữ liệu
CICIoT2023 đã được mô tả trong `data_description.md`.

Các thay đổi dữ liệu không làm thay đổi logic PerFed-SKD:

- 10 client non-IID đã được tạo trước bằng Dirichlet `α=0.2`;
- mỗi mẫu có 25 đặc trưng đã chọn bằng XGBoost và 1 trong 34 nhãn;
- mỗi client dùng 95% local train và 5% local validation;
- `global_train_data.csv` pretrain global model đúng một epoch;
- `global_test_data.csv` chỉ được dùng sau khi chọn checkpoint;
- mô hình của global server, student, historical teacher và personalized client
  đều là **{scenario["model_family"]}** trong notebook này.

Notebook chạy trên đúng **Kaggle GPU T4 ×2**, sử dụng PyTorch CUDA AMP và hai
worker client-parallel cố định. Đây không phải mô phỏng bằng cách lấy mẫu:
toàn bộ dữ liệu xác nhận được sử dụng. Runtime benchmark 1/2 CUDA streams;
GRU/Transformer giữ một stream để bảo toàn RNG dropout độc lập theo client,
còn CNN-1D có thể chọn hai stream khi throughput thực đo cao hơn.
"""


def method_markdown() -> str:
    return r"""## Phương pháp PerFed-SKD

### Mục tiêu liên kết

Với dữ liệu riêng \(D_m\) tại client \(m\), paper viết:

\[
\min_\omega F(\omega)
=\sum_{m=1}^{M}\frac{|D_m|}{|D|}F_m(\omega).
\]

Server khởi tạo có seed và pretrain \(\omega^0\) một epoch trên
`global_train_data.csv`.

### Self-knowledge distillation

Trước mỗi local update, personalized model hiện tại được đóng băng thành
historical teacher \(V_m\). Student nhận global state nếu client thuộc tập chọn
\(S_t\); nếu không, student tiếp tục từ personalized state hiện tại.

\[
\ell_{\mathrm{CE}}=\operatorname{CE}(y,z_m),
\]

\[
\ell_{\mathrm{KD}}
=T^2\operatorname{KL}\left(
\operatorname{softmax}(z_{V_m}/T)
\;\middle\|\;
\operatorname{softmax}(z_m/T)
\right),
\]

\[
\phi_m=\ell_{\mathrm{CE}}+\lambda\ell_{\mathrm{KD}},
\qquad \lambda=1,\quad T=1,
\]

\[
\omega_m\leftarrow\omega_m-\eta\nabla\phi_m,\qquad \eta=0.01.
\]

Teacher chỉ suy luận và không nhận gradient. Student dùng SGD, CUDA AMP, batch
1024 và đúng một local epoch mỗi round.

### Device selection đúng Algorithm 1

Round đầu có \(S_1=M\). Sau khi đủ 10 client local-train và được đánh giá trên
local validation, server tính:

\[
A^t=\frac{1}{|M|}\sum_{m\in M}a_m^t.
\]

Tập nhận global model ở round sau là:

\[
S_{t+1}=\{m\in M\mid a_m^t<A^t\}.
\]

\(A^t\) là trung bình accuracy cục bộ của đủ 10 thiết bị, không phải accuracy
của global model trên dữ liệu local. So sánh dùng strict `<`.

Chỉ client thuộc \(S_t\) download/upload model trong round \(t\). Server tổng
hợp không trọng số theo Algorithm 1:

\[
\omega^{t+1}=\frac{1}{|S_t|}
\sum_{m\in S_t}\omega_m^{t+1}.
\]

Mọi client vẫn tiếp tục local training để duy trì personalized knowledge và để
server có đủ \(a_m^t\). Nếu \(S_t\) rỗng, global state được giữ nguyên và
logical communication của round bằng 0.
"""


def architecture_markdown(scenario: dict) -> str:
    family = scenario["model_family"]
    if family == "gru":
        architecture = """| Stage | Shape / configuration |
|---|---|
| Input reshape | `[B,25] → [B,25,1]` |
| GRU | 2 layers, hidden 64, dropout 0.2 |
| Readout | last step `[B,64]` |
| Classifier | Linear `64 → 34` |

Exact trainable parameters: **40,034**."""
    elif family == "transformer":
        architecture = """| Stage | Shape / configuration |
|---|---|
| Scalar projection | Linear `1 → 64` per feature |
| Position | learned `[1,25,64]` |
| Encoder | 2 layers, 4 heads, FFN 128, dropout 0.1 |
| Readout | mean pooling + LayerNorm |
| Classifier | Linear `64 → 34` |

Exact trainable parameters: **71,010**."""
    else:
        architecture = """| Stage | Shape / configuration |
|---|---|
| Input reshape | `[B,25] → [B,1,25]` |
| Block 1 | Conv `1→32`, k3, BN, ReLU |
| Block 2 | Conv `32→64`, k3, BN, ReLU, MaxPool2 |
| Block 3 | Conv `64→128`, k3, BN, ReLU |
| Readout | adaptive average pool + Linear `128→34` |

Exact trainable parameters: **35,874**."""
    return f"""## Kiến trúc và hợp đồng output

{architecture}

Mọi model nhận `[batch,25]`, trả logits `[batch,34]` và không chứa softmax.
Checkpoint giữ `model_state_dict`, `personalized_model_state_dicts`,
`historical_teacher_model_state_dicts`, initial/pretrained/global hashes và
metadata cần thiết để so sánh trọng số giữa các phương án.

Output lâu dài được ghi dưới `/kaggle/working/{scenario["run_name"]}/` gồm
best/last checkpoints, log, CSV/JSON metric, confusion matrix, communication,
runtime, GPU cache/memory telemetry và bảy biểu đồ chẩn đoán.
"""


def environment_code(config: dict) -> str:
    config_literal = json.dumps(config, indent=4, ensure_ascii=False).replace(
        ": null", ": None"
    )
    return f"""from pathlib import Path
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

n_gpu = torch.cuda.device_count()
assert n_gpu == 2, f"Notebook requires exactly two GPUs, found {{n_gpu}}"
GPU_NAMES = [torch.cuda.get_device_name(index) for index in range(2)]
assert all("T4" in name for name in GPU_NAMES), (
    f"Notebook requires two NVIDIA T4 GPUs, found {{GPU_NAMES}}"
)

CONFIG = {config_literal}

INPUT_DIR = Path(CONFIG["input_dir"])
OUTPUT_DIR = Path(CONFIG["output_dir"])
TEMP_DIR = Path(CONFIG["temp_dir"])
CACHE_DIR = Path(CONFIG["cache_dir"])
MANIFEST_PATH = Path(CONFIG["manifest_path"])

for directory in [
    OUTPUT_DIR / "checkpoints",
    OUTPUT_DIR / "logs",
    OUTPUT_DIR / "metrics",
    OUTPUT_DIR / "artifacts",
    TEMP_DIR,
    CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPUs:", GPU_NAMES)
print("Run:", CONFIG["run_name"])
"""


def preprocessing_code() -> str:
    return r"""import logging
import math

PIPELINE_STARTED = time.perf_counter()
FEATURE_COLUMNS = CONFIG["feature_columns"]
LABEL_COLUMN = CONFIG["label_column"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]
CSV_CHUNK_ROWS = 262144

required_inputs = [
    INPUT_DIR / f"client_{client_id}_train.csv"
    for client_id in range(1, CONFIG["num_clients"] + 1)
] + [
    INPUT_DIR / "global_train_data.csv",
    INPUT_DIR / "global_test_data.csv",
    INPUT_DIR / "label_mapping.csv",
]
missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
assert not missing_inputs, f"Missing required Kaggle inputs: {missing_inputs}"

label_frame = pd.read_csv(INPUT_DIR / "label_mapping.csv")
assert list(label_frame.columns) == ["Encoded_ID", "Label_Name"]
label_frame = label_frame.sort_values("Encoded_ID").reset_index(drop=True)
assert label_frame["Encoded_ID"].tolist() == list(range(CONFIG["num_classes"]))
LABEL_MAPPING = {
    str(int(row.Encoded_ID)): str(row.Label_Name)
    for row in label_frame.itertuples(index=False)
}
assert LABEL_MAPPING[str(CONFIG["benign_label_id"])].upper() == "BENIGN"


def convert_csv_to_memmap(csv_path, cache_prefix):
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    assert header == EXPECTED_COLUMNS, (
        f"Header/order mismatch in {csv_path.name}: {header}"
    )
    feature_path = CACHE_DIR / f"{cache_prefix}_features.f32.dat"
    label_path = CACHE_DIR / f"{cache_prefix}_labels.i64.dat"
    row_count = 0
    class_counts = np.zeros(CONFIG["num_classes"], dtype=np.int64)
    with open(feature_path, "wb") as feature_handle, open(
        label_path, "wb"
    ) as label_handle:
        reader = pd.read_csv(
            csv_path,
            chunksize=CSV_CHUNK_ROWS,
            usecols=EXPECTED_COLUMNS,
            memory_map=True,
        )
        for chunk_id, chunk in enumerate(reader, start=1):
            assert list(chunk.columns) == EXPECTED_COLUMNS
            features = chunk[FEATURE_COLUMNS].to_numpy(
                dtype=np.float32, copy=True
            )
            labels_raw = chunk[LABEL_COLUMN].to_numpy(copy=True)
            assert np.isfinite(features).all(), (
                f"NaN/Inf in {csv_path.name}, chunk {chunk_id}"
            )
            assert np.isfinite(labels_raw).all()
            labels = labels_raw.astype(np.int64, copy=False)
            assert np.array_equal(labels_raw, labels), (
                f"Non-integer labels in {csv_path.name}"
            )
            assert labels.min(initial=0) >= 0
            assert labels.max(initial=0) < CONFIG["num_classes"]
            features.tofile(feature_handle)
            labels.tofile(label_handle)
            class_counts += np.bincount(
                labels, minlength=CONFIG["num_classes"]
            )
            row_count += len(labels)
            if chunk_id % 20 == 0:
                print(
                    f"{csv_path.name}: converted "
                    f"{row_count:,} rows"
                )
    assert row_count > 0, f"Empty dataset: {csv_path}"
    expected_feature_bytes = row_count * CONFIG["num_features"] * 4
    expected_label_bytes = row_count * 8
    assert feature_path.stat().st_size == expected_feature_bytes
    assert label_path.stat().st_size == expected_label_bytes
    return {
        "row_count": row_count,
        "features": {
            "path": str(feature_path),
            "dtype": "float32",
            "shape": [row_count, CONFIG["num_features"]],
        },
        "labels": {
            "path": str(label_path),
            "dtype": "int64",
            "shape": [row_count],
        },
        "class_counts": class_counts.tolist(),
    }


def stratified_local_split(dataset_meta, client_id):
    labels = np.memmap(
        dataset_meta["labels"]["path"],
        dtype=np.int64,
        mode="r",
        shape=tuple(dataset_meta["labels"]["shape"]),
    )
    train_parts = []
    validation_parts = []
    for class_id in range(CONFIG["num_classes"]):
        positions = np.flatnonzero(labels == class_id).astype(
            np.int64, copy=False
        )
        if len(positions) == 0:
            continue
        class_rng = np.random.default_rng(
            CONFIG["seed"] + client_id * 1009 + class_id * 9176
        )
        class_rng.shuffle(positions)
        if len(positions) == 1:
            validation_count = 0
        else:
            validation_count = max(
                1,
                int(round(len(positions) * CONFIG["validation_fraction"])),
            )
            validation_count = min(validation_count, len(positions) - 1)
        validation_parts.append(positions[:validation_count])
        train_parts.append(positions[validation_count:])
    train_indices = np.concatenate(train_parts).astype(np.int64, copy=False)
    validation_indices = (
        np.concatenate(validation_parts).astype(np.int64, copy=False)
        if validation_parts
        else np.empty(0, dtype=np.int64)
    )
    split_rng = np.random.default_rng(CONFIG["seed"] + client_id * 65537)
    split_rng.shuffle(train_indices)
    split_rng.shuffle(validation_indices)
    assert len(train_indices) + len(validation_indices) == len(labels)
    split_membership = np.zeros(len(labels), dtype=np.uint8)
    split_membership[train_indices] = 1
    assert not split_membership[validation_indices].any()
    split_membership[validation_indices] = 1
    assert int(split_membership.sum()) == len(labels)
    del split_membership
    assert train_indices.min(initial=0) >= 0
    assert train_indices.max(initial=0) < len(labels)
    assert validation_indices.min(initial=0) >= 0
    assert validation_indices.max(initial=0) < len(labels)
    train_path = CACHE_DIR / f"client_{client_id}_train_indices.npy"
    validation_path = CACHE_DIR / f"client_{client_id}_validation_indices.npy"
    np.save(train_path, train_indices)
    np.save(validation_path, validation_indices)
    dataset_meta.update(
        {
            "train_indices_path": str(train_path),
            "validation_indices_path": str(validation_path),
            "train_examples": int(len(train_indices)),
            "validation_examples": int(len(validation_indices)),
        }
    )
    return dataset_meta


DATASETS = {"clients": {}}
distribution_rows = []
for client_id in range(1, CONFIG["num_clients"] + 1):
    print(f"Converting client {client_id}")
    client_meta = convert_csv_to_memmap(
        INPUT_DIR / f"client_{client_id}_train.csv",
        f"client_{client_id}",
    )
    client_meta = stratified_local_split(client_meta, client_id)
    DATASETS["clients"][str(client_id)] = client_meta
    for class_id, count in enumerate(client_meta["class_counts"]):
        distribution_rows.append(
            {
                "client_id": client_id,
                "class_id": class_id,
                "label": LABEL_MAPPING[str(class_id)],
                "examples": int(count),
            }
        )

print("Converting global_train_data.csv for central pretraining")
DATASETS["global_train"] = convert_csv_to_memmap(
    INPUT_DIR / "global_train_data.csv", "global_train"
)
print("Converting global_test_data.csv")
DATASETS["global_test"] = convert_csv_to_memmap(
    INPUT_DIR / "global_test_data.csv", "global_test"
)

client_total_rows = sum(
    meta["row_count"] for meta in DATASETS["clients"].values()
)
client_train_rows = sum(
    meta["train_examples"] for meta in DATASETS["clients"].values()
)
client_validation_rows = sum(
    meta["validation_examples"] for meta in DATASETS["clients"].values()
)
assert client_total_rows == DATASETS["global_train"]["row_count"], (
    "The sum of client rows must equal global_train_data.csv rows"
)

DATASET_SUMMARY = {
    "feature_count": CONFIG["num_features"],
    "class_count": CONFIG["num_classes"],
    "client_count": CONFIG["num_clients"],
    "client_total_rows": client_total_rows,
    "client_train_rows": client_train_rows,
    "client_validation_rows": client_validation_rows,
    "global_pretrain_rows": DATASETS["global_train"]["row_count"],
    "global_test_rows": DATASETS["global_test"]["row_count"],
    "client_row_counts": {
        client_id: DATASETS["clients"][client_id]["row_count"]
        for client_id in DATASETS["clients"]
    },
    "client_train_counts": {
        client_id: DATASETS["clients"][client_id]["train_examples"]
        for client_id in DATASETS["clients"]
    },
    "client_validation_counts": {
        client_id: DATASETS["clients"][client_id]["validation_examples"]
        for client_id in DATASETS["clients"]
    },
    "client_class_counts": {
        client_id: DATASETS["clients"][client_id]["class_counts"]
        for client_id in DATASETS["clients"]
    },
    "feature_columns": FEATURE_COLUMNS,
    "label_mapping": LABEL_MAPPING,
}

preprocessing_seconds = time.perf_counter() - PIPELINE_STARTED
MANIFEST = {
    "config": CONFIG,
    "datasets": DATASETS,
    "label_mapping": LABEL_MAPPING,
    "dataset_summary": DATASET_SUMMARY,
    "preprocessing_seconds": preprocessing_seconds,
}
with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
    json.dump(MANIFEST, handle, indent=2, ensure_ascii=False)
with open(
    OUTPUT_DIR / "metrics" / "dataset_summary.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(DATASET_SUMMARY, handle, indent=2, ensure_ascii=False)
with open(
    OUTPUT_DIR / "metrics" / "config.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(CONFIG, handle, indent=2, ensure_ascii=False)
pd.DataFrame(distribution_rows).to_csv(
    OUTPUT_DIR / "metrics" / "client_class_distribution.csv",
    index=False,
)

notebook_logger = logging.getLogger("notebook_preprocessing")
notebook_logger.setLevel(logging.INFO)
notebook_handler = logging.FileHandler(
    OUTPUT_DIR / "logs" / "run.log", mode="w", encoding="utf-8"
)
notebook_logger.addHandler(notebook_handler)
notebook_logger.info(
    "Preprocessing complete: %s client rows, %s test rows, %.2f seconds",
    f"{client_total_rows:,}",
    f"{DATASETS['global_test']['row_count']:,}",
    preprocessing_seconds,
)
notebook_handler.flush()
notebook_handler.close()
notebook_logger.removeHandler(notebook_handler)
logging.shutdown()

print(json.dumps(DATASET_SUMMARY, indent=2, ensure_ascii=False))
print(f"Preprocessing seconds: {preprocessing_seconds:.2f}")
"""


def runtime_cell(runtime_source: str) -> str:
    return (
        "CLIENT_PARALLEL_SCRIPT_PATH = TEMP_DIR / "
        'f"{CONFIG[\'run_name\']}_client_parallel.py"\n'
        "CLIENT_PARALLEL_RUNTIME_SOURCE = r'''\n"
        + runtime_source
        + "\n'''\n"
        "CLIENT_PARALLEL_SCRIPT_PATH.write_text(\n"
        "    CLIENT_PARALLEL_RUNTIME_SOURCE, encoding=\"utf-8\"\n"
        ")\n"
        "print(f\"Wrote runtime: {CLIENT_PARALLEL_SCRIPT_PATH}\")\n"
    )


def launcher_code() -> str:
    return r"""import subprocess

logging.shutdown()
launch_started = time.perf_counter()
command = [sys.executable, str(CLIENT_PARALLEL_SCRIPT_PATH)]
launch_environment = os.environ.copy()
launch_environment.update(
    {
        "TRAINING_MANIFEST": str(MANIFEST_PATH),
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
)
completed = subprocess.run(
    command,
    check=True,
    env=launch_environment,
)
subprocess_wall_seconds = time.perf_counter() - launch_started
assert completed.returncode == 0

runtime_path = OUTPUT_DIR / "metrics" / "runtime_breakdown.json"
with open(runtime_path, "r", encoding="utf-8") as handle:
    runtime_breakdown = json.load(handle)
runtime_breakdown["subprocess_wall_seconds"] = subprocess_wall_seconds
runtime_breakdown["total_notebook_pipeline_seconds"] = (
    runtime_breakdown["preprocessing_seconds"] + subprocess_wall_seconds
)
with open(runtime_path, "w", encoding="utf-8") as handle:
    json.dump(runtime_breakdown, handle, indent=2, ensure_ascii=False)

summary_path = OUTPUT_DIR / "metrics" / "summary.json"
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
summary["runtime_breakdown"] = runtime_breakdown
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)

print(f"Subprocess wall time: {subprocess_wall_seconds:.2f} seconds")
"""


def verification_code() -> str:
    return r"""from IPython.display import Image, display

required_relative_paths = [
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "logs/run.log",
    "logs/rank_1.log",
    "metrics/config.json",
    "metrics/dataset_summary.json",
    "metrics/client_class_distribution.csv",
    "metrics/history_pretrain.csv",
    "metrics/history_pretrain.json",
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
]
missing_or_empty = [
    relative
    for relative in required_relative_paths
    if not (OUTPUT_DIR / relative).is_file()
    or (OUTPUT_DIR / relative).stat().st_size == 0
]
assert not missing_or_empty, f"Missing/empty outputs: {missing_or_empty}"

history_pretrain = pd.read_csv(OUTPUT_DIR / "metrics" / "history_pretrain.csv")
history_round = pd.read_csv(OUTPUT_DIR / "metrics" / "history_round.csv")
history_client = pd.read_csv(OUTPUT_DIR / "metrics" / "history_client.csv")
history_local_epoch = pd.read_csv(
    OUTPUT_DIR / "metrics" / "history_local_epoch.csv"
)
assert len(history_pretrain) == 1
assert len(history_round) == CONFIG["communication_rounds"]
assert len(history_client) == (
    CONFIG["communication_rounds"] * CONFIG["num_clients"]
)
assert len(history_local_epoch) == (
    CONFIG["communication_rounds"] * CONFIG["num_clients"]
)
assert history_client.groupby("round")["client_id"].nunique().eq(10).all()
assert history_local_epoch["sampler_padding_rows"].eq(0).all()

with open(
    OUTPUT_DIR / "metrics" / "summary.json", "r", encoding="utf-8"
) as handle:
    final_summary = json.load(handle)
assert final_summary["status"] == "completed"
print(json.dumps(final_summary["final_global_test"], indent=2))

for artifact_name in [
    "class_distribution.png",
    "accuracy_f1_curves.png",
    "loss_curves.png",
    "confusion_matrix.png",
    "per_class_f1.png",
    "runtime_per_round.png",
    "communication_cumulative.png",
]:
    display(Image(filename=str(OUTPUT_DIR / "artifacts" / artifact_name)))
"""


def build_notebook(scenario: dict, runtime_source: str) -> dict:
    config = config_for(scenario)
    cells = [
        markdown_cell(introduction_markdown(scenario)),
        markdown_cell(method_markdown()),
        markdown_cell(architecture_markdown(scenario)),
        code_cell(environment_code(config)),
        markdown_cell(
            "## Tiền xử lý và cache\n\n"
            "Cell tiếp theo xác thực toàn bộ hợp đồng dữ liệu, chuyển CSV sang "
            "memmap dùng một lần và tạo local split có seed. Không lấy mẫu dữ liệu."
        ),
        code_cell(preprocessing_code()),
        markdown_cell(
            "## Runtime client-parallel\n\n"
            "Runtime độc lập được ghi vào `/kaggle/temp`, sau đó notebook đóng "
            "logger và chạy nó bằng Python subprocess. Coordinator không tạo "
            "CUDA model trước khi spawn hai worker."
        ),
        code_cell(runtime_cell(runtime_source)),
        code_cell(launcher_code()),
        markdown_cell(
            "## Nghiệm thu và hiển thị output\n\n"
            "Cell cuối kiểm tra file không rỗng, exact history row counts và "
            "render toàn bộ plot trong Kaggle."
        ),
        code_cell(verification_code()),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [
                    {
                        "sourceId": "odixe0502/data-for-10clients",
                        "sourceType": "datasetVersion",
                    }
                ],
                "dockerImageVersionId": None,
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    runtime_source = RUNTIME_TEMPLATE_PATH.read_text(encoding="utf-8")
    for scenario in SCENARIOS:
        folder = ROOT / scenario["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        notebook_path = folder / f"{scenario['run_name']}.ipynb"
        notebook = build_notebook(scenario, runtime_source)
        notebook_path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print(notebook_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
