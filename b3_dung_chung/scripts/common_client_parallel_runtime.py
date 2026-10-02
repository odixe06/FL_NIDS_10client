from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast


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
LABEL_COLUMN = "Label"
NUM_CLASSES = 34
METRIC_NAMES = [
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
]
MODEL_PARAMETER_COUNTS = {"gru": 40034, "transformer": 71010, "cnn1d": 35874}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def configure_logger(path: Path, name: str) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(message)s"
    )
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def seed_from(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def forward(self, features):
        sequence, _ = self.gru(features.unsqueeze(-1))
        return self.classifier(sequence[:, -1])


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_projection = nn.Linear(1, 64)
        self.position = nn.Parameter(torch.empty(1, len(FEATURE_COLUMNS), 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, NUM_CLASSES)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, features):
        tokens = self.scalar_projection(features.unsqueeze(-1))
        tokens = tokens + self.position[:, : features.shape[1]]
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded.mean(dim=1)))


class CNN1DClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, features):
        embedding = self.network(features.unsqueeze(1)).squeeze(-1)
        return self.classifier(embedding)


class PFEDESProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 1, kernel_size=3, padding=1)

    def forward(self, features):
        values = F.relu(self.conv1(features.unsqueeze(1)))
        return self.conv2(values).squeeze(1)


def build_classifier(family: str) -> nn.Module:
    if family == "gru":
        return GRUClassifier()
    if family == "transformer":
        return TransformerClassifier()
    if family == "cnn1d":
        return CNN1DClassifier()
    raise ValueError(f"Unsupported model family: {family}")


def initial_state(module_factory, seed: int) -> OrderedDict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = module_factory()
    return cpu_state(module)


def cpu_state(module: nn.Module) -> OrderedDict:
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in module.state_dict().items()
    )


def clone_state(state) -> OrderedDict:
    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state.items())


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def validate_model_contract(family: str) -> None:
    model = build_classifier(family)
    count = parameter_count(model)
    if count != MODEL_PARAMETER_COUNTS[family]:
        raise RuntimeError(f"Unexpected {family} parameter count: {count}")
    with torch.inference_mode():
        output = model(torch.zeros(2, len(FEATURE_COLUMNS)))
    if tuple(output.shape) != (2, NUM_CLASSES):
        raise RuntimeError(f"Unexpected {family} output shape: {tuple(output.shape)}")
    proxy = PFEDESProxy()
    if parameter_count(proxy) != 57:
        raise RuntimeError("Unexpected pFedES proxy parameter count")


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"Invalid confusion shape: {matrix.shape}")
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(true_positive + false_negative) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    total = float(matrix.sum())
    tp_sum = float(true_positive.sum())
    fp_sum = float(false_positive.sum())
    fn_sum = float(false_negative.sum())
    micro_precision = tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum else 0.0
    micro_recall = tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    weighted_denominator = float(support.sum())
    result = {
        "accuracy": tp_sum / total if total else 0.0,
        "macro_precision": float(precision.mean()),
        "micro_precision": float(micro_precision),
        "weighted_precision": (
            float(np.sum(precision * support) / weighted_denominator)
            if weighted_denominator
            else 0.0
        ),
        "macro_recall": float(recall.mean()),
        "micro_recall": float(micro_recall),
        "weighted_recall": (
            float(np.sum(recall * support) / weighted_denominator)
            if weighted_denominator
            else 0.0
        ),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(micro_f1),
        "weighted_f1": (
            float(np.sum(f1 * support) / weighted_denominator)
            if weighted_denominator
            else 0.0
        ),
    }
    for name in METRIC_NAMES:
        if not np.isfinite(result[name]) or not 0.0 <= result[name] <= 1.0:
            raise RuntimeError(f"Invalid metric {name}: {result[name]}")
    for name in ("micro_precision", "micro_recall", "micro_f1"):
        if not math.isclose(result[name], result["accuracy"], abs_tol=1e-12):
            raise RuntimeError(f"{name} does not equal accuracy for single-label data")
    return result


def count_csv_rows(path: Path) -> int:
    line_count = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            line_count += block.count(b"\n")
    if line_count < 1:
        raise ValueError(f"CSV has no header: {path}")
    return line_count - 1


def validate_header(path: Path) -> None:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    expected = FEATURE_COLUMNS + [LABEL_COLUMN]
    if columns != expected:
        raise ValueError(f"Unexpected header for {path.name}: {columns}")


def convert_csv(path: Path, prefix: Path, chunk_rows: int) -> dict:
    validate_header(path)
    rows = count_csv_rows(path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    feature_path = prefix.with_name(prefix.name + "_features.npy")
    label_path = prefix.with_name(prefix.name + "_labels.npy")
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, len(FEATURE_COLUMNS)),
    )
    labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.int64,
        shape=(rows,),
    )
    offset = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    reader = pd.read_csv(path, chunksize=chunk_rows)
    for frame in reader:
        if frame.columns.tolist() != FEATURE_COLUMNS + [LABEL_COLUMN]:
            raise ValueError(f"Header changed while reading {path}")
        feature_values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        label_values = frame[LABEL_COLUMN].to_numpy(dtype=np.int64, copy=True)
        if not np.isfinite(feature_values).all():
            raise ValueError(f"NaN/Inf features in {path}")
        if label_values.size and (label_values.min() < 0 or label_values.max() >= NUM_CLASSES):
            raise ValueError(f"Label outside [0,{NUM_CLASSES - 1}] in {path}")
        end = offset + len(frame)
        features[offset:end] = feature_values
        labels[offset:end] = label_values
        class_counts += np.bincount(label_values, minlength=NUM_CLASSES)
        offset = end
    features.flush()
    labels.flush()
    if offset != rows or int(class_counts.sum()) != rows:
        raise RuntimeError(f"CSV accounting failed for {path}")
    return {
        "features_path": str(feature_path),
        "labels_path": str(label_path),
        "rows": rows,
        "class_counts": class_counts.tolist(),
        "source_path": str(path),
    }


def make_perfed_split(dataset: dict, client_id: int, config: dict) -> dict:
    labels = np.load(dataset["labels_path"], mmap_mode="r")
    rng = np.random.default_rng(seed_from(config["seed"], "perfed_split", client_id))
    train_parts = []
    validation_parts = []
    for class_id in range(NUM_CLASSES):
        positions = np.flatnonzero(labels == class_id).astype(np.int64, copy=False)
        if len(positions) <= 1:
            train_parts.append(positions)
            continue
        rng.shuffle(positions)
        validation_count = int(round(len(positions) * 0.10))
        validation_count = min(max(validation_count, 1), len(positions) - 1)
        validation_parts.append(positions[:validation_count])
        train_parts.append(positions[validation_count:])
    train_indices = np.concatenate(train_parts) if train_parts else np.empty(0, np.int64)
    validation_indices = (
        np.concatenate(validation_parts) if validation_parts else np.empty(0, np.int64)
    )
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    prefix = Path(dataset["features_path"]).with_suffix("")
    train_path = prefix.with_name(prefix.name + "_train_idx.npy")
    validation_path = prefix.with_name(prefix.name + "_validation_idx.npy")
    np.save(train_path, train_indices)
    np.save(validation_path, validation_indices)
    validate_client_split_accounting(
        dataset["rows"], train_indices, validation_indices
    )
    dataset.update(
        {
            "train_indices_path": str(train_path),
            "validation_indices_path": str(validation_path),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
        }
    )
    return dataset


def validate_client_split_accounting(
    source_rows: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> None:
    if len(train_indices) + len(validation_indices) != source_rows:
        raise RuntimeError("PerFed-SKD split is not exhaustive")
    joined = np.concatenate([train_indices, validation_indices])
    if len(np.unique(joined)) != source_rows:
        raise RuntimeError("PerFed-SKD split contains duplicate or missing rows")
    if joined.size and (joined.min() < 0 or joined.max() >= source_rows):
        raise RuntimeError("PerFed-SKD split index outside source range")


def prepare_datasets(config: dict, logger: logging.Logger) -> dict:
    input_dir = Path(config["input_dir"])
    cache_dir = Path(config["cache_dir"])
    required = [input_dir / f"client_{client_id}_train.csv" for client_id in range(1, 11)]
    required.extend(
        [
            input_dir / "global_train_data.csv",
            input_dir / "global_test_data.csv",
            input_dir / "label_mapping.csv",
        ]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    clients = {}
    for client_id in range(1, 11):
        dataset = convert_csv(
            input_dir / f"client_{client_id}_train.csv",
            cache_dir / f"client_{client_id}",
            config["csv_chunk_rows"],
        )
        if config["method"] == "perfed_skd":
            dataset = make_perfed_split(dataset, client_id, config)
        else:
            dataset.update(
                {
                    "train_indices_path": None,
                    "validation_indices_path": None,
                    "train_rows": int(dataset["rows"]),
                    "validation_rows": 0,
                }
            )
        clients[str(client_id)] = dataset
        logger.info(
            "Prepared client %s: train=%s validation=%s",
            client_id,
            f"{dataset['train_rows']:,}",
            f"{dataset['validation_rows']:,}",
        )
    global_test = convert_csv(
        input_dir / "global_test_data.csv",
        cache_dir / "global_test",
        config["csv_chunk_rows"],
    )
    global_train_rows = count_csv_rows(input_dir / "global_train_data.csv")
    validate_header(input_dir / "global_train_data.csv")
    global_train = None
    if config["method"] == "perfed_skd":
        global_train = convert_csv(
            input_dir / "global_train_data.csv",
            cache_dir / "global_train",
            config["csv_chunk_rows"],
        )
    label_frame = pd.read_csv(input_dir / "label_mapping.csv")
    if label_frame.columns.tolist() != ["Encoded_ID", "Label_Name"]:
        raise ValueError("Unexpected label_mapping.csv header")
    label_mapping = {
        int(row.Encoded_ID): str(row.Label_Name)
        for row in label_frame.itertuples(index=False)
    }
    if sorted(label_mapping) != list(range(NUM_CLASSES)):
        raise ValueError("label_mapping.csv must contain IDs 0..33 exactly")
    return {
        "clients": clients,
        "global_test": global_test,
        "global_train": global_train,
        "global_train_rows": global_train_rows,
        "label_mapping": label_mapping,
    }


def validate_numpy_classification_cache(features, labels) -> None:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Invalid cached feature shape: {features.shape}")
    if features.dtype != np.float32 or labels.dtype != np.int64:
        raise ValueError("Invalid cached classification dtypes")
    if len(features) != len(labels):
        raise ValueError("Cached feature/label row mismatch")
    if len(labels) and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError("Cached labels outside [0,33]")


def copy_numpy_to_gpu(array: np.ndarray, device: torch.device, chunk_rows: int):
    target = torch.empty(array.shape, dtype=torch.from_numpy(np.empty((), dtype=array.dtype)).dtype, device=device)
    for start in range(0, len(array), chunk_rows):
        end = min(start + chunk_rows, len(array))
        source = torch.from_numpy(np.array(array[start:end], copy=True))
        target[start:end].copy_(source, non_blocking=False)
    return target


def dataset_bytes(meta: dict, include_indices: bool) -> int:
    total = int(meta["rows"]) * (len(FEATURE_COLUMNS) * 4 + 8)
    if include_indices:
        total += int(meta["train_rows"]) * 8 + int(meta["validation_rows"]) * 8
    return total


def load_gpu_dataset(meta: dict, device: torch.device, config: dict) -> dict:
    features = np.load(meta["features_path"], mmap_mode="r")
    labels = np.load(meta["labels_path"], mmap_mode="r")
    validate_numpy_classification_cache(features, labels)
    entry = {
        "mode": "full_gpu_cache",
        "features": copy_numpy_to_gpu(features, device, config["gpu_copy_chunk_rows"]),
        "labels": copy_numpy_to_gpu(labels, device, config["gpu_copy_chunk_rows"]),
        "rows": int(meta["rows"]),
        "train_rows": int(meta.get("train_rows", meta["rows"])),
        "validation_rows": int(meta.get("validation_rows", 0)),
        "meta": meta,
    }
    if meta.get("train_indices_path"):
        train_indices = np.load(meta["train_indices_path"], mmap_mode="r")
        validation_indices = np.load(meta["validation_indices_path"], mmap_mode="r")
        entry["train_indices"] = copy_numpy_to_gpu(
            train_indices, device, config["gpu_copy_chunk_rows"]
        )
        entry["validation_indices"] = copy_numpy_to_gpu(
            validation_indices, device, config["gpu_copy_chunk_rows"]
        )
    else:
        entry["train_indices"] = None
        entry["validation_indices"] = None
    return entry


class MemmapRows(torch.utils.data.Dataset):
    def __init__(self, meta: dict, indices_path: str | None = None):
        self.features = np.load(meta["features_path"], mmap_mode="r")
        self.labels = np.load(meta["labels_path"], mmap_mode="r")
        validate_numpy_classification_cache(self.features, self.labels)
        self.indices = (
            np.load(indices_path, mmap_mode="r") if indices_path else None
        )

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.labels)

    def __getitem__(self, index):
        row = int(self.indices[index]) if self.indices is not None else index
        return np.asarray(self.features[row], dtype=np.float32), np.int64(self.labels[row])


def make_cpu_loader(
    meta: dict,
    split: str,
    config: dict,
    shuffle: bool,
    seed: int,
):
    indices_path = None
    if split == "train":
        indices_path = meta.get("train_indices_path")
    elif split == "validation":
        indices_path = meta.get("validation_indices_path")
        if not indices_path:
            raise ValueError("Validation requested without validation indices")
    dataset = MemmapRows(meta, indices_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["per_client_batch_size"] if split == "train" else config["evaluation_batch_size"],
        shuffle=shuffle,
        num_workers=config["dataloader_workers_per_gpu"],
        pin_memory=True,
        persistent_workers=config["dataloader_workers_per_gpu"] > 0,
        generator=generator,
        drop_last=False,
    )


def initialize_worker_cache(
    client_ids: list[int],
    datasets: dict,
    device: torch.device,
    config: dict,
) -> tuple[dict, dict, dict]:
    total_vram = int(torch.cuda.get_device_properties(device).total_memory)
    cache_budget_bytes = int(total_vram * config["gpu_cache_fraction"])
    include_indices = config["method"] == "perfed_skd"
    planned = dataset_bytes(datasets["global_test"], False)
    planned += sum(
        dataset_bytes(datasets["clients"][str(client_id)], include_indices)
        for client_id in client_ids
    )
    entries = {}
    if planned <= cache_budget_bytes:
        for client_id in client_ids:
            entries[client_id] = load_gpu_dataset(
                datasets["clients"][str(client_id)], device, config
            )
        test_entry = load_gpu_dataset(datasets["global_test"], device, config)
        mode = "full_gpu_cache"
        actual = planned
    else:
        for client_id in client_ids:
            meta = datasets["clients"][str(client_id)]
            features = np.load(meta["features_path"], mmap_mode="r")
            labels = np.load(meta["labels_path"], mmap_mode="r")
            validate_numpy_classification_cache(features, labels)
            entries[client_id] = {"mode": "pinned_memory_fallback", "meta": meta}
        test_meta = datasets["global_test"]
        test_features = np.load(test_meta["features_path"], mmap_mode="r")
        test_labels = np.load(test_meta["labels_path"], mmap_mode="r")
        validate_numpy_classification_cache(test_features, test_labels)
        test_entry = {"mode": "pinned_memory_fallback", "meta": test_meta}
        mode = "pinned_memory_fallback"
        actual = 0
    report = {
        "mode": mode,
        "cache_budget_bytes": cache_budget_bytes,
        "planned_cache_bytes": planned,
        "actual_cache_bytes": actual,
        "cache_hits": 0,
        "cache_misses": 0 if mode == "full_gpu_cache" else len(client_ids) + 1,
        "cache_evictions": 0,
        "fallback_used": mode != "full_gpu_cache",
    }
    return entries, test_entry, report


def entry_train_rows(entry: dict) -> int:
    # Never write this as entry.get("train_rows", entry["meta"]["train_rows"]):
    # dict.get evaluates its default eagerly, and the server-pretrain metadata
    # (global_train) carries only "rows" because it is never split into a local
    # train/validation pair.
    if "train_rows" in entry:
        return int(entry["train_rows"])
    meta = entry["meta"]
    return int(meta.get("train_rows", meta["rows"]))


def entry_validation_rows(entry: dict) -> int:
    # The pinned-memory fallback entry keeps only {"mode", "meta"}, so the row
    # counts have to be read back from the metadata in that mode.
    if "validation_rows" in entry:
        return int(entry["validation_rows"])
    return int(entry["meta"].get("validation_rows", 0))


def training_batches(
    entry: dict,
    config: dict,
    device: torch.device,
    client_id: int,
    round_index: int,
    phase: str,
    stream: torch.cuda.Stream,
):
    train_examples = entry_train_rows(entry)
    if entry["mode"] == "full_gpu_cache":
        context = make_training_context(
            client_id,
            stream,
            device,
            train_examples,
            seed_from(config["seed"], phase, round_index, client_id),
        )
        order = context["order"]
        for start in range(0, train_examples, config["per_client_batch_size"]):
            end = min(start + config["per_client_batch_size"], train_examples)
            positions = order[start:end]
            if entry["train_indices"] is not None:
                rows = entry["train_indices"].index_select(0, positions)
            else:
                rows = positions
            yield (
                entry["features"].index_select(0, rows),
                entry["labels"].index_select(0, rows),
                context,
            )
    else:
        loader = make_cpu_loader(
            entry["meta"],
            "train",
            config,
            True,
            seed_from(config["seed"], phase, round_index, client_id),
        )
        context = {
            "loss_sums": torch.zeros(8, dtype=torch.float64, device=device),
            "order": None,
        }
        for cpu_features, cpu_labels in loader:
            yield (
                cpu_features.to(device, non_blocking=True),
                cpu_labels.to(device, non_blocking=True),
                context,
            )


def make_training_context(
    client_id: int,
    stream: torch.cuda.Stream,
    device: torch.device,
    train_examples: int,
    seed: int,
) -> dict:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    with torch.cuda.stream(stream):
        order = torch.randperm(train_examples, generator=generator, device=device)
        loss_sums = torch.zeros(8, dtype=torch.float64, device=device)
    return {
        "client_id": client_id,
        "stream": stream,
        "order": order,
        "loss_sums": loss_sums,
    }


def evaluate_entry(
    model: nn.Module,
    entry: dict,
    config: dict,
    device: torch.device,
    split: str,
    row_start: int | None = None,
    row_end: int | None = None,
) -> dict:
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    examples = 0
    with torch.inference_mode():
        if entry["mode"] == "full_gpu_cache":
            if split == "validation":
                indices = entry["validation_indices"]
                batches = (
                    (
                        entry["features"].index_select(0, indices[start:end]),
                        entry["labels"].index_select(0, indices[start:end]),
                    )
                    for start in range(0, len(indices), config["evaluation_batch_size"])
                    for end in [
                        min(start + config["evaluation_batch_size"], len(indices))
                    ]
                )
            else:
                start_value = 0 if row_start is None else row_start
                end_value = entry["rows"] if row_end is None else row_end
                if not 0 <= start_value <= end_value <= entry["rows"]:
                    raise ValueError("Invalid contiguous evaluation shard")
                # Global-test shards are already contiguous. Direct slicing avoids
                # allocating a multi-million-row CUDA arange and an index_select
                # gather for every personalized model in every round.
                batches = (
                    (entry["features"][start:end], entry["labels"][start:end])
                    for start in range(
                        start_value, end_value, config["evaluation_batch_size"]
                    )
                    for end in [
                        min(start + config["evaluation_batch_size"], end_value)
                    ]
                )
            for features, labels in batches:
                with autocast("cuda"):
                    logits = model(features)
                    loss = F.cross_entropy(logits, labels, reduction="sum")
                predictions = logits.argmax(dim=1)
                encoded = labels.to(torch.int64) * NUM_CLASSES + predictions
                confusion += torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
                    NUM_CLASSES, NUM_CLASSES
                )
                loss_sum += loss.to(torch.float64)
                examples += len(labels)
        else:
            meta = entry["meta"]
            if split == "validation":
                loader = make_cpu_loader(meta, "validation", config, False, config["seed"])
                skip_rows = 0
                take_rows = int(meta["validation_rows"])
            else:
                loader = make_cpu_loader(meta, "test", config, False, config["seed"])
                skip_rows = 0 if row_start is None else row_start
                take_rows = int(meta["rows"]) if row_end is None else row_end - skip_rows
            consumed = 0
            skipped = 0
            for cpu_features, cpu_labels in loader:
                if skipped + len(cpu_labels) <= skip_rows:
                    skipped += len(cpu_labels)
                    continue
                offset = max(0, skip_rows - skipped)
                available = min(len(cpu_labels) - offset, take_rows - consumed)
                if available <= 0:
                    break
                features = cpu_features[offset : offset + available].to(device, non_blocking=True)
                labels = cpu_labels[offset : offset + available].to(device, non_blocking=True)
                with autocast("cuda"):
                    logits = model(features)
                    loss = F.cross_entropy(logits, labels, reduction="sum")
                predictions = logits.argmax(dim=1)
                encoded = labels.to(torch.int64) * NUM_CLASSES + predictions
                confusion += torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
                    NUM_CLASSES, NUM_CLASSES
                )
                loss_sum += loss.to(torch.float64)
                examples += available
                consumed += available
                skipped += len(cpu_labels)
                if consumed >= take_rows:
                    break
    torch.cuda.synchronize(device)
    matrix = confusion.detach().cpu().numpy()
    return {
        "confusion": matrix,
        "examples": int(examples),
        "loss_sum": float(loss_sum.detach().cpu().numpy()),
        "metrics": metrics_from_confusion(matrix),
    }


def train_fd_ids_client(
    client_id: int,
    entry: dict,
    global_state: dict,
    family: str,
    config: dict,
    device: torch.device,
    round_index: int,
) -> dict:
    model = build_classifier(family).to(device)
    model.load_state_dict(global_state, strict=True)
    teacher = build_classifier(family).to(device)
    teacher.load_state_dict(global_state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], foreach=True
    )
    scaler = GradScaler("cuda")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    loss_sums = None
    optimizer_steps = 0
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for features, labels, context in training_batches(
            entry, config, device, client_id, round_index, "fd_ids", stream
        ):
            if loss_sums is None:
                loss_sums = context["loss_sums"]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                student_logits = model(features)
                with torch.no_grad():
                    teacher_logits = teacher(features)
                hard_loss = F.cross_entropy(student_logits, labels)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / config["temperature"], dim=1),
                    F.softmax(teacher_logits / config["temperature"], dim=1),
                    reduction="batchmean",
                ) * (config["temperature"] ** 2)
                proximal_sum = torch.zeros((), dtype=torch.float32, device=device)
                for local_parameter, global_parameter in zip(
                    model.parameters(), teacher.parameters()
                ):
                    proximal_sum = proximal_sum + torch.sum(
                        (local_parameter - global_parameter) ** 2
                    )
                proximal_loss = 0.5 * config["fedprox_mu"] * proximal_sum
                total_loss = (
                    config["hard_loss_weight"] * hard_loss
                    + (1.0 - config["hard_loss_weight"]) * soft_loss
                    + config["proximal_weight"] * proximal_loss
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_examples = len(labels)
            loss_sums[:4] += torch.stack(
                [hard_loss, soft_loss, proximal_loss, total_loss]
            ).detach().to(torch.float64) * batch_examples
            loss_sums[4] += batch_examples
            optimizer_steps += 1
    stream.synchronize()
    values = loss_sums.detach().cpu().numpy()
    denominator = max(float(values[4]), 1.0)
    return {
        "client_id": client_id,
        "train_examples": entry_train_rows(entry),
        "optimizer_steps": optimizer_steps,
        "hard_loss": float(values[0] / denominator),
        "soft_loss": float(values[1] / denominator),
        "proximal_loss": float(values[2] / denominator),
        "total_loss": float(values[3] / denominator),
        "train_seconds": time.perf_counter() - started,
        "model_state": cpu_state(model),
    }


def train_perfed_client(
    client_id: int,
    entry: dict,
    global_state: dict,
    personalized_state: dict,
    selected: bool,
    family: str,
    config: dict,
    device: torch.device,
    round_index: int,
) -> dict:
    teacher = build_classifier(family).to(device)
    teacher.load_state_dict(personalized_state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = build_classifier(family).to(device)
    model.load_state_dict(global_state if selected else personalized_state, strict=True)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        foreach=True,
    )
    scaler = GradScaler("cuda")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    loss_sums = None
    optimizer_steps = 0
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for features, labels, context in training_batches(
            entry, config, device, client_id, round_index, "perfed_skd", stream
        ):
            if loss_sums is None:
                loss_sums = context["loss_sums"]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(features)
                with torch.no_grad():
                    teacher_logits = teacher(features)
                cross_entropy = F.cross_entropy(logits, labels)
                distillation = F.kl_div(
                    F.log_softmax(logits / config["temperature"], dim=1),
                    F.softmax(teacher_logits / config["temperature"], dim=1),
                    reduction="batchmean",
                ) * (config["temperature"] ** 2)
                total_loss = cross_entropy + config["distillation_lambda"] * distillation
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_examples = len(labels)
            loss_sums[:3] += torch.stack(
                [cross_entropy, distillation, total_loss]
            ).detach().to(torch.float64) * batch_examples
            loss_sums[3] += batch_examples
            optimizer_steps += 1
    stream.synchronize()
    validation = evaluate_entry(model, entry, config, device, "validation")
    values = loss_sums.detach().cpu().numpy()
    denominator = max(float(values[3]), 1.0)
    return {
        "client_id": client_id,
        "selected": bool(selected),
        "train_examples": entry_train_rows(entry),
        "validation_examples": entry_validation_rows(entry),
        "optimizer_steps": optimizer_steps,
        "cross_entropy_loss": float(values[0] / denominator),
        "distillation_kl_loss": float(values[1] / denominator),
        "total_loss": float(values[2] / denominator),
        "train_seconds": time.perf_counter() - started,
        "model_state": cpu_state(model),
        "validation": validation,
    }


def train_proxymodel_client(
    client_id: int,
    entry: dict,
    global_proxy_state: dict,
    personalized_state: dict,
    family: str,
    config: dict,
    device: torch.device,
    round_index: int,
) -> dict:
    model = build_classifier(family).to(device)
    model.load_state_dict(personalized_state, strict=True)
    proxy = CNN1DClassifier().to(device)
    proxy.load_state_dict(global_proxy_state, strict=True)
    model.train()
    proxy.train()
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), proxy.parameters()),
        lr=config["learning_rate"],
        foreach=True,
    )
    scaler = GradScaler("cuda")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    loss_sums = None
    optimizer_steps = 0
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for features, labels, context in training_batches(
            entry, config, device, client_id, round_index, "proxymodel", stream
        ):
            if loss_sums is None:
                loss_sums = context["loss_sums"]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                personal_logits = model(features)
                proxy_logits = proxy(features)
                personal_loss = F.cross_entropy(personal_logits, labels)
                proxy_loss = F.cross_entropy(proxy_logits, labels)
                discrepancy = F.mse_loss(
                    personal_logits.softmax(dim=1), proxy_logits.softmax(dim=1)
                )
                distillation = discrepancy / (
                    personal_loss.detach()
                    + proxy_loss.detach()
                    + config["adaptive_epsilon"]
                )
                total_loss = personal_loss + proxy_loss + distillation
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_examples = len(labels)
            loss_sums[:4] += torch.stack(
                [personal_loss, proxy_loss, distillation, total_loss]
            ).detach().to(torch.float64) * batch_examples
            loss_sums[4] += batch_examples
            optimizer_steps += 1
    stream.synchronize()
    finetune_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    finetune_examples = 0
    if (
        round_index == config["rounds"]
        and config["final_personalized_finetune_epochs"] == 1
    ):
        finetune_optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], foreach=True
        )
        finetune_scaler = GradScaler("cuda")
        model.train()
        with torch.cuda.stream(stream):
            for features, labels, _ in training_batches(
                entry,
                config,
                device,
                client_id,
                round_index,
                "proxymodel_final_finetune",
                stream,
            ):
                finetune_optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    finetune_loss = F.cross_entropy(model(features), labels)
                finetune_scaler.scale(finetune_loss).backward()
                finetune_scaler.step(finetune_optimizer)
                finetune_scaler.update()
                finetune_loss_sum += (
                    finetune_loss.detach().to(torch.float64) * len(labels)
                )
                finetune_examples += len(labels)
        stream.synchronize()
    values = loss_sums.detach().cpu().numpy()
    denominator = max(float(values[4]), 1.0)
    return {
        "client_id": client_id,
        "train_examples": entry_train_rows(entry),
        "optimizer_steps": optimizer_steps,
        "personalized_label_loss": float(values[0] / denominator),
        "proxy_label_loss": float(values[1] / denominator),
        "adaptive_distillation_loss": float(values[2] / denominator),
        "total_loss": float(values[3] / denominator),
        "final_finetune_loss": (
            float(finetune_loss_sum.detach().cpu().numpy())
            / max(finetune_examples, 1)
        ),
        "train_seconds": time.perf_counter() - started,
        "model_state": cpu_state(model),
        "local_proxy_state": cpu_state(proxy),
    }


def train_pfedes_client(
    client_id: int,
    entry: dict,
    global_proxy_state: dict,
    personalized_state: dict,
    family: str,
    config: dict,
    device: torch.device,
    round_index: int,
) -> dict:
    model = build_classifier(family).to(device)
    model.load_state_dict(personalized_state, strict=True)
    proxy = PFEDESProxy().to(device)
    proxy.load_state_dict(global_proxy_state, strict=True)
    local_optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], foreach=True
    )
    proxy_optimizer = torch.optim.SGD(
        proxy.parameters(), lr=config["learning_rate"], foreach=True
    )
    local_scaler = GradScaler("cuda")
    proxy_scaler = GradScaler("cuda")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    local_sums = None
    proxy_sums = None
    local_steps = 0
    proxy_steps = 0
    started = time.perf_counter()
    model.train()
    proxy.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in proxy.parameters():
        parameter.requires_grad_(False)
    with torch.cuda.stream(stream):
        for features, labels, context in training_batches(
            entry, config, device, client_id, round_index, "pfedes_local", stream
        ):
            if local_sums is None:
                local_sums = context["loss_sums"]
            local_optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                with torch.no_grad():
                    enhanced = proxy(features)
                enhanced_loss = F.cross_entropy(model(enhanced), labels)
                original_loss = F.cross_entropy(model(features), labels)
                total_loss = (
                    config["enhanced_loss_weight"] * enhanced_loss
                    + (1.0 - config["enhanced_loss_weight"]) * original_loss
                )
            local_scaler.scale(total_loss).backward()
            local_scaler.step(local_optimizer)
            local_scaler.update()
            batch_examples = len(labels)
            local_sums[:3] += torch.stack(
                [enhanced_loss, original_loss, total_loss]
            ).detach().to(torch.float64) * batch_examples
            local_sums[3] += batch_examples
            local_steps += 1
    stream.synchronize()
    local_optimizer.zero_grad(set_to_none=True)
    model.train()
    proxy.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in proxy.parameters():
        parameter.requires_grad_(True)
    with torch.cuda.stream(stream):
        for features, labels, context in training_batches(
            entry, config, device, client_id, round_index, "pfedes_proxy", stream
        ):
            if proxy_sums is None:
                proxy_sums = context["loss_sums"]
            proxy_optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                proxy_loss = F.cross_entropy(model(proxy(features)), labels)
            proxy_scaler.scale(proxy_loss).backward()
            proxy_scaler.step(proxy_optimizer)
            proxy_scaler.update()
            batch_examples = len(labels)
            proxy_sums[0] += proxy_loss.detach().to(torch.float64) * batch_examples
            proxy_sums[1] += batch_examples
            proxy_steps += 1
    stream.synchronize()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    local_values = local_sums.detach().cpu().numpy()
    proxy_values = proxy_sums.detach().cpu().numpy()
    local_denominator = max(float(local_values[3]), 1.0)
    proxy_denominator = max(float(proxy_values[1]), 1.0)
    return {
        "client_id": client_id,
        "train_examples": entry_train_rows(entry),
        "local_optimizer_steps": local_steps,
        "proxy_optimizer_steps": proxy_steps,
        "enhanced_cross_entropy_loss": float(local_values[0] / local_denominator),
        "original_cross_entropy_loss": float(local_values[1] / local_denominator),
        "local_total_loss": float(local_values[2] / local_denominator),
        "proxy_cross_entropy_loss": float(proxy_values[0] / proxy_denominator),
        "train_seconds": time.perf_counter() - started,
        "model_state": cpu_state(model),
        "local_proxy_state": cpu_state(proxy),
    }


def aggregate_weighted(states: list[dict], weights: list[int]) -> OrderedDict:
    if not states or len(states) != len(weights):
        raise ValueError("Invalid weighted aggregation inputs")
    denominator = float(sum(weights))
    result = OrderedDict()
    for name in states[0]:
        tensors = [state[name] for state in states]
        if tensors[0].is_floating_point():
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float64)
            for tensor, weight in zip(tensors, weights):
                accumulator += tensor.to(torch.float64) * (weight / denominator)
            result[name] = accumulator.to(tensors[0].dtype)
        else:
            result[name] = tensors[0].clone()
    return result


def aggregate_unweighted(states: list[dict], fallback: dict) -> OrderedDict:
    if not states:
        return clone_state(fallback)
    return aggregate_weighted(states, [1] * len(states))


def benchmark_stream_candidate(
    family: str,
    method: str,
    stream_count: int,
    config: dict,
    device: torch.device,
) -> dict:
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    contexts = []
    batch_size = config["per_client_batch_size"]
    for stream_index, stream in enumerate(streams):
        model = build_classifier(family).to(device).train()
        auxiliary = None
        if method == "proxymodel":
            auxiliary = CNN1DClassifier().to(device).train()
        elif method == "pfedes":
            auxiliary = PFEDESProxy().to(device).train()
        elif method == "perfed_skd":
            auxiliary = build_classifier(family).to(device).eval()
            for parameter in auxiliary.parameters():
                parameter.requires_grad_(False)
        parameters = list(model.parameters())
        if auxiliary is not None and method in {"proxymodel", "pfedes"}:
            parameters.extend(auxiliary.parameters())
        optimizer = torch.optim.SGD(parameters, lr=0.01, foreach=True)
        scaler = GradScaler("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_from(config["seed"], "benchmark", stream_index))
        with torch.cuda.stream(stream):
            features = torch.randn(
                batch_size, len(FEATURE_COLUMNS), device=device, generator=generator
            )
            labels = torch.randint(
                0, NUM_CLASSES, (batch_size,), device=device, generator=generator
            )
        contexts.append((stream, model, auxiliary, optimizer, scaler, features, labels))

    def enqueue_step(context):
        stream, model, auxiliary, optimizer, scaler, features, labels = context
        with torch.cuda.stream(stream):
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                if method == "pfedes":
                    enhanced = auxiliary(features)
                    loss = (
                        F.cross_entropy(model(features), labels)
                        + F.cross_entropy(model(enhanced.detach()), labels)
                        + F.cross_entropy(model(enhanced), labels)
                    )
                elif method == "proxymodel":
                    personal_logits = model(features)
                    proxy_logits = auxiliary(features)
                    loss = (
                        F.cross_entropy(personal_logits, labels)
                        + F.cross_entropy(proxy_logits, labels)
                        + F.mse_loss(
                            personal_logits.softmax(dim=1),
                            proxy_logits.softmax(dim=1),
                        )
                    )
                elif method == "perfed_skd":
                    student_logits = model(features)
                    with torch.no_grad():
                        teacher_logits = auxiliary(features)
                    loss = F.cross_entropy(student_logits, labels) + F.kl_div(
                        F.log_softmax(student_logits, dim=1),
                        F.softmax(teacher_logits, dim=1),
                        reduction="batchmean",
                    )
                else:
                    loss = F.cross_entropy(model(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    for _ in range(config["stream_benchmark_warmup_steps"]):
        for context in contexts:
            enqueue_step(context)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(config["stream_benchmark_steps"]):
        for context in contexts:
            enqueue_step(context)
    end.record()
    torch.cuda.synchronize(device)
    seconds = start.elapsed_time(end) / 1000.0
    samples = batch_size * config["stream_benchmark_steps"] * stream_count
    return {
        "stream_count": stream_count,
        "seconds": seconds,
        "samples_per_second": samples / max(seconds, 1e-12),
        "seconds_per_step": seconds
        / max(config["stream_benchmark_steps"] * stream_count, 1),
    }


def choose_client_assignment(
    train_rows: dict[int, int],
    seconds_per_step_by_gpu: dict[int, float],
    batch_size: int,
) -> tuple[dict[int, list[int]], dict]:
    client_ids = tuple(sorted(train_rows))
    steps = {
        client_id: math.ceil(train_rows[client_id] / batch_size)
        for client_id in client_ids
    }
    best_key = None
    best_assignment = None
    for mask in range(1, 2 ** len(client_ids) - 1):
        left = tuple(
            client_id
            for index, client_id in enumerate(client_ids)
            if mask & (1 << index)
        )
        right = tuple(client_id for client_id in client_ids if client_id not in left)
        # Do not canonicalize (left, right): GPU 0 and GPU 1 can benchmark at
        # different rates, so the mirrored placement is a distinct candidate.
        load0 = sum(steps[client_id] for client_id in left) * seconds_per_step_by_gpu[0]
        load1 = sum(steps[client_id] for client_id in right) * seconds_per_step_by_gpu[1]
        key = (max(load0, load1), abs(load0 - load1), left, right)
        if best_key is None or key < best_key:
            best_key = key
            best_assignment = (left, right)
    assignment = {0: list(best_assignment[0]), 1: list(best_assignment[1])}
    predicted = {
        "gpu_0_seconds": sum(steps[c] for c in assignment[0]) * seconds_per_step_by_gpu[0],
        "gpu_1_seconds": sum(steps[c] for c in assignment[1]) * seconds_per_step_by_gpu[1],
    }
    return assignment, predicted


def train_server_pretrain(
    state: dict,
    dataset_meta: dict,
    family: str,
    config: dict,
    device: torch.device,
) -> dict:
    entry = load_gpu_dataset(dataset_meta, device, config)
    entry["train_rows"] = int(dataset_meta["rows"])
    entry["train_indices"] = None
    model = build_classifier(family).to(device)
    model.load_state_dict(state, strict=True)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], foreach=True
    )
    scaler = GradScaler("cuda")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    examples = 0
    steps = 0
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for features, labels, _ in training_batches(
            entry, config, device, 0, 0, "server_pretrain", stream
        ):
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = F.cross_entropy(model(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().to(torch.float64) * len(labels)
            examples += len(labels)
            steps += 1
    stream.synchronize()
    return {
        "state": cpu_state(model),
        "train_examples": examples,
        "optimizer_steps": steps,
        "cross_entropy_loss": float(loss_sum.detach().cpu().numpy()) / max(examples, 1),
        "seconds": time.perf_counter() - started,
    }


def worker_main(gpu_id, task_queue, result_queue, manifest_path):
    logger = None
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        config = manifest["config"]
        output_dir = Path(config["output_dir"])
        logger = configure_logger(
            output_dir / "logs" / f"worker_{gpu_id}.log", f"worker_{gpu_id}"
        )
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
        logger.info("Worker %s started on %s", gpu_id, device)
        seed_everything(seed_from(config["seed"], "worker", gpu_id))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        client_entries = {}
        test_entry = None
        owned_clients = []
        cache_report = None
        while True:
            task = task_queue.get()
            task_id = task["task_id"]
            kind = task["kind"]
            logger.info("Starting task %s (%s)", task_id, kind)
            if kind == "shutdown":
                logger.info("Worker %s shutting down", gpu_id)
                result_queue.put(
                    {"kind": "shutdown_ok", "task_id": task_id, "gpu_id": gpu_id}
                )
                break
            if kind == "benchmark":
                rows = [
                    benchmark_stream_candidate(
                        config["model_family"],
                        config["method"],
                        stream_count,
                        config,
                        device,
                    )
                    for stream_count in config["stream_candidates"]
                ]
                result_queue.put(
                    {"kind": "task_ok", "task_id": task_id, "gpu_id": gpu_id, "payload": rows}
                )
                continue
            if kind == "pretrain":
                seed_everything(
                    seed_from(
                        config["seed"], "server_pretrain", config["model_family"]
                    )
                )
                result = train_server_pretrain(
                    task["state"],
                    manifest["datasets"]["global_train"],
                    config["model_family"],
                    config,
                    device,
                )
                payload_path = Path(config["temp_dir"]) / f"{task_id}.pt"
                torch.save(result, payload_path)
                result_queue.put(
                    {
                        "kind": "task_ok",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                    }
                )
                continue
            if kind == "initialize":
                owned_clients = list(task["client_ids"])
                client_entries, test_entry, cache_report = initialize_worker_cache(
                    owned_clients, manifest["datasets"], device, config
                )
                result_queue.put(
                    {
                        "kind": "task_ok",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload": cache_report,
                    }
                )
                continue
            if kind == "train_round":
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                def train_owned_client(client_id: int) -> dict:
                    torch.cuda.set_device(gpu_id)
                    if config["model_family"] in {"gru", "transformer"}:
                        # Stochastic families run one client at a time. Resetting
                        # their dropout RNG per client makes benchmark activity
                        # and assignment changes unable to perturb training.
                        seed_everything(
                            seed_from(
                                config["seed"],
                                config["method"],
                                config["model_family"],
                                task["round"],
                                client_id,
                            )
                        )
                    entry = client_entries[client_id]
                    if config["method"] == "fd_ids_noniid":
                        return train_fd_ids_client(
                            client_id,
                            entry,
                            task["server_state"],
                            config["model_family"],
                            config,
                            device,
                            task["round"],
                        )
                    elif config["method"] == "perfed_skd":
                        return train_perfed_client(
                            client_id,
                            entry,
                            task["server_state"],
                            task["personalized_states"][str(client_id)],
                            client_id in task["selected_clients"],
                            config["model_family"],
                            config,
                            device,
                            task["round"],
                        )
                    elif config["method"] == "proxymodel":
                        return train_proxymodel_client(
                            client_id,
                            entry,
                            task["server_state"],
                            task["personalized_states"][str(client_id)],
                            config["model_family"],
                            config,
                            device,
                            task["round"],
                        )
                    elif config["method"] == "pfedes":
                        return train_pfedes_client(
                            client_id,
                            entry,
                            task["server_state"],
                            task["personalized_states"][str(client_id)],
                            config["model_family"],
                            config,
                            device,
                            task["round"],
                        )
                    else:
                        raise ValueError(f"Unsupported method: {config['method']}")
                runtime_stream_count = int(task.get("runtime_stream_count", 1))
                if cache_report["mode"] != "full_gpu_cache":
                    runtime_stream_count = 1
                if runtime_stream_count == 2:
                    if config["model_family"] != "cnn1d":
                        raise RuntimeError("Two-stream client overlap is allowed only for deterministic CNN-1D workloads")
                    scheduled_clients = sorted(
                        owned_clients,
                        key=lambda client_id: (
                            -entry_train_rows(client_entries[client_id]),
                            client_id,
                        ),
                    )
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        client_results = list(
                            pool.map(train_owned_client, scheduled_clients)
                        )
                else:
                    scheduled_clients = list(owned_clients)
                    client_results = [train_owned_client(client_id) for client_id in owned_clients]
                payload_path = Path(config["temp_dir"]) / f"{task_id}.pt"
                torch.save(
                    {
                        "clients": client_results,
                        "scheduled_clients": scheduled_clients,
                        "runtime_stream_count": runtime_stream_count,
                        "worker_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    },
                    payload_path,
                )
                result_queue.put(
                    {
                        "kind": "task_ok",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                    }
                )
                continue
            if kind == "evaluate_clients":
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                evaluations = []
                for client_id in task["client_ids"]:
                    model = build_classifier(config["model_family"]).to(device)
                    model.load_state_dict(
                        task["client_states"][str(client_id)], strict=True
                    )
                    evaluations.append(
                        {
                            "client_id": client_id,
                            "test": evaluate_entry(
                                model, test_entry, config, device, "test"
                            ),
                        }
                    )
                payload_path = Path(config["temp_dir"]) / f"{task_id}.pt"
                torch.save(
                    {
                        "clients": evaluations,
                        "worker_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    },
                    payload_path,
                )
                result_queue.put(
                    {
                        "kind": "task_ok",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                    }
                )
                continue
            if kind == "evaluate_server":
                family = "cnn1d" if config["method"] == "proxymodel" else config["model_family"]
                model = build_classifier(family).to(device)
                model.load_state_dict(task["server_state"], strict=True)
                result = evaluate_entry(
                    model,
                    test_entry,
                    config,
                    device,
                    "test",
                    task["row_start"],
                    task["row_end"],
                )
                payload_path = Path(config["temp_dir"]) / f"{task_id}.pt"
                torch.save(result, payload_path)
                result_queue.put(
                    {
                        "kind": "task_ok",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                    }
                )
                continue
            raise ValueError(f"Unknown worker task: {kind}")
    except Exception:
        error = traceback.format_exc()
        if logger is not None:
            logger.exception("Worker failed")
        result_queue.put(
            {"kind": "worker_error", "gpu_id": gpu_id, "traceback": error}
        )
        raise


def collect_results(result_queue, workers, expected_ids: list[str], timeout: int) -> dict:
    pending = set(expected_ids)
    results = {}
    deadline = time.monotonic() + timeout
    while pending:
        for worker in workers:
            if worker.exitcode not in (None, 0):
                raise RuntimeError(f"Worker {worker.pid} exited with {worker.exitcode}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for tasks: {sorted(pending)}")
        try:
            message = result_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            continue
        if message.get("kind") == "worker_error":
            raise RuntimeError(message["traceback"])
        task_id = message.get("task_id")
        if task_id not in pending:
            raise RuntimeError(f"Duplicate or unexpected task result: {task_id}")
        results[task_id] = message
        pending.remove(task_id)
    return results


def load_message(message):
    if "payload" in message:
        return message["payload"]
    path = Path(message["payload_path"])
    value = torch.load(path, map_location="cpu", weights_only=False)
    path.unlink(missing_ok=True)
    return value


def combine_evaluations(evaluations: list[dict]) -> dict:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    examples = 0
    loss_sum = 0.0
    for evaluation in evaluations:
        confusion += np.asarray(evaluation["confusion"], dtype=np.int64)
        examples += int(evaluation["examples"])
        loss_sum += float(evaluation["loss_sum"])
    return {
        "confusion": confusion,
        "examples": examples,
        "loss_sum": loss_sum,
        "metrics": metrics_from_confusion(confusion),
    }


def save_table(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(json_safe(rows)).to_csv(csv_path, index=False)
    atomic_json(json_path, rows)


def evaluation_row(
    config: dict,
    round_index: int,
    model_scope: str,
    client_id: int | None,
    checkpoint_relative_path: str,
    evaluation: dict,
) -> dict:
    if int(evaluation["confusion"].sum()) != int(evaluation["examples"]):
        raise RuntimeError("Evaluation confusion accounting mismatch")
    return {
        "method": config["method"],
        "scenario": config["scenario"],
        "run_name": config["run_name"],
        "round": round_index,
        "model_scope": model_scope,
        "client_id": client_id,
        "checkpoint_relative_path": checkpoint_relative_path,
        "test_examples": int(evaluation["examples"]),
        **{name: float(evaluation["metrics"][name]) for name in METRIC_NAMES},
    }


def save_confusion(
    output_dir: Path,
    round_index: int,
    scope: str,
    client_id: int | None,
    confusion: np.ndarray,
) -> str:
    name = "server.npy" if scope == "server" else f"client_{client_id:02d}.npy"
    path = (
        output_dir
        / "metrics"
        / "confusion_matrices"
        / f"round_{round_index:03d}"
        / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(confusion, dtype=np.int64))
    return str(path.relative_to(output_dir))


def save_round_checkpoints(
    output_dir: Path,
    config: dict,
    datasets: dict,
    round_index: int,
    server_state: dict,
    personalized_states: dict[int, dict],
    selected_current: list[int],
    selected_next: list[int],
    manifest_rows: list[dict],
) -> dict:
    round_dir = output_dir / "checkpoints" / f"round_{round_index:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "round": round_index,
        "method": config["method"],
        "scenario": config["scenario"],
        "config": config,
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": datasets["label_mapping"],
        "selected_clients_current_round": selected_current,
        "selected_clients_next_round": selected_next,
    }
    server_checkpoint = {
        **common,
        "model_scope": "server",
        "client_id": None,
        "server_state_dict": clone_state(server_state),
        "state_hash": state_hash(server_state),
        "server_state_type": config["server_state_type"],
    }
    server_path = round_dir / "server.pt"
    atomic_torch_save(server_checkpoint, server_path)
    manifest_rows.append(
        {
            "round": round_index,
            "model_scope": "server",
            "client_id": None,
            "path": str(server_path.relative_to(output_dir)),
            "state_hash": server_checkpoint["state_hash"],
            "state_type": config["server_state_type"],
            "classification_status": (
                "applicable" if config["server_classifier_available"] else "not_applicable"
            ),
            "evaluation_metric_lookup": (
                f"metrics/evaluation_metrics.csv#round={round_index}&model_scope=server"
                if config["server_classifier_available"]
                else None
            ),
            "optimizer_state_persistence": "optimizer_recreated_for_each_local_update",
        }
    )
    client_paths = {}
    if config["persistent_personalized_clients"]:
        for client_id in range(1, config["num_clients"] + 1):
            state = personalized_states[client_id]
            client_checkpoint = {
                **common,
                "model_scope": "client",
                "client_id": client_id,
                "model_state_dict": clone_state(state),
                "state_hash": state_hash(state),
                "model_family": config["model_family"],
            }
            client_path = round_dir / f"client_{client_id:02d}.pt"
            atomic_torch_save(client_checkpoint, client_path)
            client_paths[client_id] = str(client_path.relative_to(output_dir))
            manifest_rows.append(
                {
                    "round": round_index,
                    "model_scope": "client",
                    "client_id": client_id,
                    "path": client_paths[client_id],
                    "state_hash": client_checkpoint["state_hash"],
                    "state_type": "persistent_personalized_classifier",
                    "classification_status": "applicable",
                    "evaluation_metric_lookup": (
                        f"metrics/evaluation_metrics.csv#round={round_index}&model_scope=client&client_id={client_id}"
                    ),
                    "optimizer_state_persistence": "optimizer_recreated_for_each_local_update",
                }
            )
    return {
        "server": str(server_path.relative_to(output_dir)),
        "clients": client_paths,
    }


def verify_checkpoint_manifest(output_dir: Path, config: dict, rows: list[dict]) -> None:
    expected_per_round = 1 + (
        config["num_clients"] if config["persistent_personalized_clients"] else 0
    )
    expected = config["rounds"] * expected_per_round
    if len(rows) != expected:
        raise RuntimeError(f"Checkpoint manifest row count {len(rows)} != {expected}")
    seen = set()
    for row in rows:
        key = (row["round"], row["model_scope"], row["client_id"])
        if key in seen:
            raise RuntimeError(f"Duplicate checkpoint manifest entry: {key}")
        seen.add(key)
        path = output_dir / row["path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing checkpoint: {path}")


def verify_evaluation_metrics(rows: list[dict], config: dict, global_test_rows: int) -> None:
    if config["evaluation_scope"] == "server_only":
        entities = 1
    elif config["evaluation_scope"] == "clients_only":
        entities = config["num_clients"]
    elif config["evaluation_scope"] == "server_and_clients":
        entities = config["num_clients"] + 1
    else:
        raise ValueError(f"Unknown evaluation scope: {config['evaluation_scope']}")
    expected = config["rounds"] * entities
    if len(rows) != expected:
        raise RuntimeError(f"Evaluation row count {len(rows)} != {expected}")
    for row in rows:
        if int(row["test_examples"]) != global_test_rows:
            raise RuntimeError("Global-test row accounting mismatch")
        for name in METRIC_NAMES:
            value = float(row[name])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"Invalid {name}: {value}")


def write_dataset_outputs(output_dir: Path, datasets: dict, config: dict) -> None:
    rows = []
    for client_id in range(1, config["num_clients"] + 1):
        meta = datasets["clients"][str(client_id)]
        for class_id, count in enumerate(meta["class_counts"]):
            rows.append(
                {
                    "client_id": client_id,
                    "class_id": class_id,
                    "count": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "metrics" / "client_class_distribution.csv", index=False
    )
    atomic_json(
        output_dir / "metrics" / "dataset_summary.json",
        {
            "split_policy": config["split_policy"],
            "clients": {
                client_id: {
                    "source_rows": meta["rows"],
                    "train_rows": meta["train_rows"],
                    "validation_rows": meta["validation_rows"],
                    "class_counts": meta["class_counts"],
                }
                for client_id, meta in datasets["clients"].items()
            },
            "global_train_rows": datasets["global_train_rows"],
            "global_test_rows": datasets["global_test"]["rows"],
            "label_mapping": datasets["label_mapping"],
        },
    )


def save_plots(
    output_dir: Path,
    evaluation_rows: list[dict],
    history_round: list[dict],
    datasets: dict,
    gpu_samples: list[dict],
) -> None:
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    client_totals = [
        datasets["clients"][str(client_id)]["rows"] for client_id in range(1, 11)
    ]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(range(1, 11), client_totals)
    axis.set_xlabel("Client")
    axis.set_ylabel("Rows")
    figure.tight_layout()
    figure.savefig(artifacts / "class_distribution.png", dpi=160)
    plt.close(figure)

    frame = pd.DataFrame(evaluation_rows)
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    for axis, metric in zip(axes.flat, METRIC_NAMES):
        for (scope, client_id), group in frame.groupby(["model_scope", "client_id"], dropna=False):
            label = scope if pd.isna(client_id) else f"client_{int(client_id):02d}"
            axis.plot(group["round"], group[metric], alpha=0.7, label=label)
        axis.set_title(metric)
        axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(artifacts / "evaluation_metric_curves.png", dpi=160)
    plt.close(figure)

    round_frame = pd.DataFrame(history_round)
    figure, axis = plt.subplots(figsize=(9, 4))
    loss_columns = [column for column in round_frame.columns if column.endswith("_loss")]
    for column in loss_columns:
        axis.plot(round_frame["round"], round_frame[column], label=column)
    if loss_columns:
        axis.legend(fontsize=7)
    axis.set_xlabel("Round")
    figure.tight_layout()
    figure.savefig(artifacts / "loss_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(round_frame["round"], round_frame["round_seconds"])
    axis.set_xlabel("Round")
    axis.set_ylabel("Seconds")
    figure.tight_layout()
    figure.savefig(artifacts / "runtime_per_round.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(round_frame["round"], round_frame["communication_mib_cumulative"])
    axis.set_xlabel("Round")
    axis.set_ylabel("Cumulative MiB")
    figure.tight_layout()
    figure.savefig(artifacts / "communication_cumulative.png", dpi=160)
    plt.close(figure)

    sample_frame = pd.DataFrame(gpu_samples)
    figure, axis = plt.subplots(figsize=(9, 4))
    if not sample_frame.empty:
        for gpu_id, group in sample_frame.groupby("gpu_id"):
            axis.plot(group["seconds"], group["utilization_percent"], label=f"GPU {gpu_id}")
        axis.legend()
    axis.set_xlabel("Pipeline seconds")
    axis.set_ylabel("GPU utilization %")
    figure.tight_layout()
    figure.savefig(artifacts / "gpu_utilization.png", dpi=160)
    plt.close(figure)


def gpu_monitor(stop_event: threading.Event, rows: list[dict], started: float) -> None:
    while not stop_event.wait(5.0):
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in completed.stdout.splitlines():
                gpu_id, utilization, memory_used, memory_total = [
                    value.strip() for value in line.split(",")
                ]
                rows.append(
                    {
                        "seconds": time.perf_counter() - started,
                        "gpu_id": int(gpu_id),
                        "utilization_percent": float(utilization),
                        "memory_used_mib": float(memory_used),
                        "memory_total_mib": float(memory_total),
                    }
                )
        except Exception:
            continue


def coordinate(workers, task_queues, result_queue, manifest: dict) -> None:
    config = manifest["config"]
    datasets = manifest["datasets"]
    output_dir = Path(config["output_dir"])
    logger = configure_logger(output_dir / "logs" / "run.log", "coordinator")
    pipeline_started = time.perf_counter()
    gpu_samples = []
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=gpu_monitor,
        args=(monitor_stop, gpu_samples, pipeline_started),
        daemon=True,
    )
    monitor.start()

    validate_model_contract(config["model_family"])
    personal_initial = initial_state(
        lambda: build_classifier(config["model_family"]), config["initialization_seed"]
    )
    if config["method"] in {"fd_ids_noniid", "perfed_skd"}:
        server_state = clone_state(personal_initial)
    elif config["method"] == "proxymodel":
        server_state = initial_state(CNN1DClassifier, config["initialization_seed"])
    elif config["method"] == "pfedes":
        server_state = initial_state(PFEDESProxy, config["initialization_seed"])
    else:
        raise ValueError(f"Unsupported method: {config['method']}")

    # Measure both physical devices before assigning any long-running work.
    # The benchmark is also used to place PerFed-SKD's inherently sequential
    # server pretrain on the faster of the two T4s.
    benchmark_ids = []
    for gpu_id in range(2):
        task_id = f"benchmark_gpu_{gpu_id}"
        benchmark_ids.append(task_id)
        task_queues[gpu_id].put({"task_id": task_id, "kind": "benchmark"})
    benchmark_messages = collect_results(
        result_queue, workers, benchmark_ids, config["worker_timeout_seconds"]
    )
    benchmark_results = {
        gpu_id: load_message(benchmark_messages[f"benchmark_gpu_{gpu_id}"])
        for gpu_id in range(2)
    }

    pretrain_history = []
    if config["method"] == "perfed_skd":
        pretrain_gpu_id = min(
            range(2),
            key=lambda gpu_id: next(
                row
                for row in benchmark_results[gpu_id]
                if row["stream_count"] == 1
            )["seconds_per_step"],
        )
        task_queues[pretrain_gpu_id].put(
            {"task_id": "server_pretrain", "kind": "pretrain", "state": server_state}
        )
        message = collect_results(
            result_queue, workers, ["server_pretrain"], config["worker_timeout_seconds"]
        )["server_pretrain"]
        pretrain = load_message(message)
        server_state = pretrain.pop("state")
        pretrain["gpu_id"] = pretrain_gpu_id
        personal_initial = clone_state(server_state)
        pretrain_history.append(pretrain)
    else:
        pretrain_history.append(
            {
                "enabled": False,
                "train_examples": 0,
                "optimizer_steps": 0,
                "cross_entropy_loss": 0.0,
                "seconds": 0.0,
            }
        )
    save_table(
        pretrain_history,
        output_dir / "metrics" / "history_pretrain.csv",
        output_dir / "metrics" / "history_pretrain.json",
    )

    seconds_per_step = {}
    selected_streams = {}
    for gpu_id in range(2):
        one_stream = next(
            row for row in benchmark_results[gpu_id] if row["stream_count"] == 1
        )
        best = max(
            benchmark_results[gpu_id],
            key=lambda row: (row["samples_per_second"], -row["stream_count"]),
        )
        if config["model_family"] in {"gru", "transformer"}:
            selected_streams[gpu_id] = 1
        else:
            speedup = best["samples_per_second"] / max(
                one_stream["samples_per_second"], 1e-12
            )
            selected_streams[gpu_id] = (
                int(best["stream_count"])
                if speedup >= config["stream_min_speedup"]
                else 1
            )
        selected_candidate = next(
            row
            for row in benchmark_results[gpu_id]
            if row["stream_count"] == selected_streams[gpu_id]
        )
        seconds_per_step[gpu_id] = float(selected_candidate["seconds_per_step"])
    train_rows = {
        client_id: int(datasets["clients"][str(client_id)]["train_rows"])
        for client_id in range(1, config["num_clients"] + 1)
    }
    client_assignment, predicted_load = choose_client_assignment(
        train_rows, seconds_per_step, config["per_client_batch_size"]
    )
    config["client_assignment"] = {
        str(gpu_id): clients for gpu_id, clients in client_assignment.items()
    }
    config["selected_streams_by_gpu"] = {
        str(gpu_id): selected_streams[gpu_id] for gpu_id in range(2)
    }
    personalized_evaluation_assignment = {
        0: list(range(1, config["num_clients"] // 2 + 1)),
        1: list(range(config["num_clients"] // 2 + 1, config["num_clients"] + 1)),
    }
    config["personalized_evaluation_assignment"] = {
        str(gpu_id): clients
        for gpu_id, clients in personalized_evaluation_assignment.items()
    }
    config["stream_benchmark"] = benchmark_results
    atomic_json(output_dir / "metrics" / "config.json", config)

    initialize_ids = []
    for gpu_id in range(2):
        task_id = f"initialize_gpu_{gpu_id}"
        initialize_ids.append(task_id)
        task_queues[gpu_id].put(
            {
                "task_id": task_id,
                "kind": "initialize",
                "client_ids": client_assignment[gpu_id],
            }
        )
    initialization_messages = collect_results(
        result_queue, workers, initialize_ids, config["worker_timeout_seconds"]
    )
    cache_reports = {
        gpu_id: load_message(initialization_messages[f"initialize_gpu_{gpu_id}"])
        for gpu_id in range(2)
    }
    logger.info("Client assignment: %s", client_assignment)

    personalized_states = {
        client_id: clone_state(personal_initial)
        for client_id in range(1, config["num_clients"] + 1)
    }
    selected_current = list(range(1, config["num_clients"] + 1))
    evaluation_rows = []
    validation_rows = []
    history_round = []
    history_client = []
    checkpoint_manifest_rows = []
    communication_cumulative = 0
    server_state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in server_state.values()
    )
    global_test_rows = int(datasets["global_test"]["rows"])

    for round_index in range(1, config["rounds"] + 1):
        round_started = time.perf_counter()
        task_ids = []
        for gpu_id in range(2):
            task_id = f"round_{round_index:03d}_gpu_{gpu_id}"
            task_ids.append(task_id)
            task_queues[gpu_id].put(
                {
                    "task_id": task_id,
                    "kind": "train_round",
                    "round": round_index,
                    "server_state": server_state,
                    "personalized_states": {
                        str(client_id): personalized_states[client_id]
                        for client_id in client_assignment[gpu_id]
                    },
                    "selected_clients": selected_current,
                    "runtime_stream_count": selected_streams[gpu_id],
                }
            )
        messages = collect_results(
            result_queue, workers, task_ids, config["worker_timeout_seconds"]
        )
        worker_payloads = [
            load_message(messages[f"round_{round_index:03d}_gpu_{gpu_id}"])
            for gpu_id in range(2)
        ]
        client_results = sorted(
            [row for payload in worker_payloads for row in payload["clients"]],
            key=lambda row: row["client_id"],
        )
        if [row["client_id"] for row in client_results] != list(range(1, 11)):
            raise RuntimeError("Incomplete round client results")

        if config["method"] == "fd_ids_noniid":
            server_state = aggregate_weighted(
                [row["model_state"] for row in client_results],
                [row["train_examples"] for row in client_results],
            )
            selected_next = list(range(1, 11))
        elif config["method"] == "perfed_skd":
            for row in client_results:
                personalized_states[row["client_id"]] = row["model_state"]
            selected_states = [
                personalized_states[client_id] for client_id in selected_current
            ]
            server_state = aggregate_unweighted(selected_states, server_state)
            local_accuracies = {
                row["client_id"]: row["validation"]["metrics"]["accuracy"]
                for row in client_results
            }
            threshold = float(np.mean(list(local_accuracies.values())))
            selected_next = sorted(
                client_id
                for client_id, accuracy in local_accuracies.items()
                if accuracy < threshold
            )
        elif config["method"] in {"proxymodel", "pfedes"}:
            for row in client_results:
                personalized_states[row["client_id"]] = row["model_state"]
            server_state = aggregate_weighted(
                [row["local_proxy_state"] for row in client_results],
                [row["train_examples"] for row in client_results],
            )
            selected_next = list(range(1, 11))
        else:
            raise ValueError(config["method"])

        if config["method"] == "perfed_skd":
            threshold = float(
                np.mean(
                    [row["validation"]["metrics"]["accuracy"] for row in client_results]
                )
            )
        else:
            threshold = None

        checkpoint_paths = save_round_checkpoints(
            output_dir,
            config,
            datasets,
            round_index,
            server_state,
            personalized_states,
            selected_current,
            selected_next,
            checkpoint_manifest_rows,
        )

        evaluation_worker_payloads = [
            {"clients": [], "worker_seconds": 0.0},
            {"clients": [], "worker_seconds": 0.0},
        ]
        client_evaluations = {}
        if config["persistent_personalized_clients"]:
            evaluation_task_ids = []
            for gpu_id in range(2):
                client_ids = personalized_evaluation_assignment[gpu_id]
                task_id = f"client_test_round_{round_index:03d}_gpu_{gpu_id}"
                evaluation_task_ids.append(task_id)
                task_queues[gpu_id].put(
                    {
                        "task_id": task_id,
                        "kind": "evaluate_clients",
                        "client_ids": client_ids,
                        "client_states": {
                            str(client_id): personalized_states[client_id]
                            for client_id in client_ids
                        },
                    }
                )
            evaluation_messages = collect_results(
                result_queue,
                workers,
                evaluation_task_ids,
                config["worker_timeout_seconds"],
            )
            evaluation_worker_payloads = [
                load_message(
                    evaluation_messages[
                        f"client_test_round_{round_index:03d}_gpu_{gpu_id}"
                    ]
                )
                for gpu_id in range(2)
            ]
            client_evaluations = {
                row["client_id"]: row["test"]
                for payload in evaluation_worker_payloads
                for row in payload["clients"]
            }
            if sorted(client_evaluations) != list(
                range(1, config["num_clients"] + 1)
            ):
                raise RuntimeError("Incomplete personalized global-test results")
            for client_id in range(1, config["num_clients"] + 1):
                test = client_evaluations[client_id]
                confusion_path = save_confusion(
                    output_dir,
                    round_index,
                    "client",
                    client_id,
                    test["confusion"],
                )
                row = evaluation_row(
                    config,
                    round_index,
                    "client",
                    client_id,
                    checkpoint_paths["clients"][client_id],
                    test,
                )
                row["confusion_matrix_relative_path"] = confusion_path
                evaluation_rows.append(row)

        if config["method"] == "perfed_skd":
            for result in client_results:
                validation = result["validation"]
                validation_rows.append(
                    {
                        "method": config["method"],
                        "scenario": config["scenario"],
                        "round": round_index,
                        "client_id": result["client_id"],
                        "validation_examples": validation["examples"],
                        "used_for_client_selection": True,
                        "selection_metric": "accuracy",
                        **{
                            name: float(validation["metrics"][name])
                            for name in METRIC_NAMES
                        },
                    }
                )

        if config["evaluation_scope"] in {"server_only", "server_and_clients"}:
            midpoint = global_test_rows // 2
            shard_ids = []
            for gpu_id, (start, end) in {
                0: (0, midpoint),
                1: (midpoint, global_test_rows),
            }.items():
                task_id = f"server_test_round_{round_index:03d}_gpu_{gpu_id}"
                shard_ids.append(task_id)
                task_queues[gpu_id].put(
                    {
                        "task_id": task_id,
                        "kind": "evaluate_server",
                        "server_state": server_state,
                        "row_start": start,
                        "row_end": end,
                    }
                )
            shard_messages = collect_results(
                result_queue, workers, shard_ids, config["worker_timeout_seconds"]
            )
            server_evaluation = combine_evaluations(
                [
                    load_message(
                        shard_messages[
                            f"server_test_round_{round_index:03d}_gpu_{gpu_id}"
                        ]
                    )
                    for gpu_id in range(2)
                ]
            )
            confusion_path = save_confusion(
                output_dir,
                round_index,
                "server",
                None,
                server_evaluation["confusion"],
            )
            row = evaluation_row(
                config,
                round_index,
                "server",
                None,
                checkpoint_paths["server"],
                server_evaluation,
            )
            row["confusion_matrix_relative_path"] = confusion_path
            evaluation_rows.append(row)

        round_seconds = time.perf_counter() - round_started
        if config["method"] == "perfed_skd":
            communication_round = 2 * len(selected_current) * server_state_bytes
        else:
            communication_round = 2 * config["num_clients"] * server_state_bytes
        communication_cumulative += communication_round
        numeric_losses = {}
        for key in sorted(client_results[0]):
            if key.endswith("_loss") and isinstance(client_results[0][key], (float, int)):
                numeric_losses[f"mean_{key}"] = float(
                    np.average(
                        [row[key] for row in client_results],
                        weights=[row["train_examples"] for row in client_results],
                    )
                )
        round_row = {
            "round": round_index,
            **numeric_losses,
            "selection_threshold_accuracy": threshold,
            "selected_clients_current": selected_current,
            "selected_clients_next": selected_next,
            "round_seconds": round_seconds,
            "communication_bytes_round": communication_round,
            "communication_mib_round": communication_round / (1024**2),
            "communication_bytes_cumulative": communication_cumulative,
            "communication_mib_cumulative": communication_cumulative / (1024**2),
            "predicted_gpu_0_seconds": predicted_load["gpu_0_seconds"],
            "predicted_gpu_1_seconds": predicted_load["gpu_1_seconds"],
            "actual_gpu_0_seconds": worker_payloads[0]["worker_seconds"],
            "actual_gpu_1_seconds": worker_payloads[1]["worker_seconds"],
            "worker_idle_seconds": abs(
                worker_payloads[0]["worker_seconds"]
                - worker_payloads[1]["worker_seconds"]
            ),
            "evaluation_gpu_0_seconds": evaluation_worker_payloads[0][
                "worker_seconds"
            ],
            "evaluation_gpu_1_seconds": evaluation_worker_payloads[1][
                "worker_seconds"
            ],
            "evaluation_worker_idle_seconds": abs(
                evaluation_worker_payloads[0]["worker_seconds"]
                - evaluation_worker_payloads[1]["worker_seconds"]
            ),
            "peak_allocated_gpu0_bytes": worker_payloads[0]["peak_allocated_bytes"],
            "peak_allocated_gpu1_bytes": worker_payloads[1]["peak_allocated_bytes"],
            "peak_reserved_gpu0_bytes": worker_payloads[0]["peak_reserved_bytes"],
            "peak_reserved_gpu1_bytes": worker_payloads[1]["peak_reserved_bytes"],
        }
        history_round.append(round_row)
        for result in client_results:
            history_client.append(
                {
                    "round": round_index,
                    "client_id": result["client_id"],
                    "train_examples": result["train_examples"],
                    "selected_for_global_update": result.get("selected", True),
                    "train_seconds": result["train_seconds"],
                    **{
                        key: value
                        for key, value in result.items()
                        if key.endswith("_loss") and isinstance(value, (float, int))
                    },
                }
            )
        save_table(
            evaluation_rows,
            output_dir / "metrics" / "evaluation_metrics.csv",
            output_dir / "metrics" / "evaluation_metrics.json",
        )
        save_table(
            history_round,
            output_dir / "metrics" / "history_round.csv",
            output_dir / "metrics" / "history_round.json",
        )
        save_table(
            history_client,
            output_dir / "metrics" / "history_client.csv",
            output_dir / "metrics" / "history_client.json",
        )
        if config["method"] == "perfed_skd":
            save_table(
                validation_rows,
                output_dir / "metrics" / "validation_metrics.csv",
                output_dir / "metrics" / "validation_metrics.json",
            )
        atomic_json(
            output_dir / "checkpoints" / "checkpoint_manifest.json",
            {
                "method": config["method"],
                "scenario": config["scenario"],
                "rounds": config["rounds"],
                "evaluation_scope": config["evaluation_scope"],
                "entries": checkpoint_manifest_rows,
            },
        )
        logger.info("Completed round %s/%s", round_index, config["rounds"])
        selected_current = selected_next

    final_rows = [row for row in evaluation_rows if row["round"] == config["rounds"]]
    save_table(
        final_rows,
        output_dir / "metrics" / "final_round_metrics.csv",
        output_dir / "metrics" / "final_round_metrics.json",
    )
    if config["server_classifier_available"]:
        server_status = {
            "status": "applicable",
            "reason": "server state returns 34-class logits",
            "evaluated_clients": config["num_clients"]
            if config["persistent_personalized_clients"]
            else 0,
        }
    else:
        server_status = {
            "status": "not_applicable",
            "reason": config["server_evaluation_not_applicable_reason"],
            "evaluated_clients": config["num_clients"],
        }
    atomic_json(output_dir / "metrics" / "server_evaluation_status.json", server_status)
    verify_checkpoint_manifest(output_dir, config, checkpoint_manifest_rows)
    verify_evaluation_metrics(evaluation_rows, config, global_test_rows)
    if config["method"] == "perfed_skd":
        if len(validation_rows) != config["rounds"] * config["num_clients"]:
            raise RuntimeError("PerFed-SKD validation metric row count mismatch")

    monitor_stop.set()
    monitor.join(timeout=10)
    save_plots(output_dir, evaluation_rows, history_round, datasets, gpu_samples)
    atomic_json(
        output_dir / "metrics" / "communication_costs.json",
        {
            "model_state_bytes": server_state_bytes,
            "total_bytes": communication_cumulative,
            "total_mib": communication_cumulative / (1024**2),
            "ddp_gradient_allreduce_estimate": {"enabled": False, "bytes": 0},
        },
    )
    total_seconds = time.perf_counter() - pipeline_started
    atomic_json(
        output_dir / "metrics" / "runtime_breakdown.json",
        {
            "total_pipeline_seconds": total_seconds,
            "round_seconds": {str(row["round"]): row["round_seconds"] for row in history_round},
            "cache_reports": cache_reports,
            "gpu_utilization_samples": gpu_samples,
        },
    )
    atomic_json(
        output_dir / "metrics" / "summary.json",
        {
            "status": "complete",
            "method": config["method"],
            "scenario": config["scenario"],
            "run_name": config["run_name"],
            "fixed_comparison_round": config["rounds"],
            "evaluation_scope": config["evaluation_scope"],
            "ten_metrics": METRIC_NAMES,
            "final_round_metrics": final_rows,
            "server_evaluation": server_status,
            "checkpoint_manifest": "checkpoints/checkpoint_manifest.json",
            "repeated_global_test_evaluation": True,
            "global_test_used_for_training_or_selection": False,
        },
    )
    logger.info("Pipeline complete in %.2f seconds", total_seconds)


def create_output_dirs(config: dict) -> None:
    output_dir = Path(config["output_dir"])
    for relative in (
        "checkpoints",
        "logs",
        "metrics/confusion_matrices",
        "artifacts",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    Path(config["temp_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    manifest_path = Path(os.environ["TRAINING_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    create_output_dirs(config)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(2)]
    workers = [
        context.Process(
            target=worker_main,
            args=(gpu_id, task_queues[gpu_id], result_queue, str(manifest_path)),
            name=f"gpu_worker_{gpu_id}",
        )
        for gpu_id in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        coordinate(workers, task_queues, result_queue, manifest)
    finally:
        shutdown_ids = []
        for gpu_id, worker in enumerate(workers):
            if worker.is_alive():
                task_id = f"shutdown_gpu_{gpu_id}"
                shutdown_ids.append(task_id)
                task_queues[gpu_id].put({"task_id": task_id, "kind": "shutdown"})
        if shutdown_ids:
            try:
                collect_results(result_queue, workers, shutdown_ids, 120)
            except Exception:
                pass
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
            if worker.exitcode not in (0, None):
                raise RuntimeError(f"Worker {worker.pid} exit code {worker.exitcode}")


if __name__ == "__main__":
    main()
