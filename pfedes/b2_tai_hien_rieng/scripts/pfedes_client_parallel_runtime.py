#!/usr/bin/env python3
"""Self-contained pFedES client-parallel runtime embedded in Kaggle notebooks."""

from __future__ import annotations

import csv
import hashlib
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
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


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
EXPECTED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]
REQUIRED_RELATIVE_OUTPUTS = [
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
    "metrics/personalized_test_metrics.json",
    "metrics/personalized_test_metrics.csv",
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


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_table_pair(records, csv_path, json_path):
    frame = pd.DataFrame(records)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)
    write_json(json_path, records)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def configure_logger(path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(message)s"
    )
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.propagate = False
    return logger


def derive_seed(base_seed, *parts):
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_state_dict(module):
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in module.state_dict().items()
    )


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def state_num_bytes(state):
    return int(
        sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    )


class ProxyFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 1, kernel_size=3, padding=1)

    def forward(self, features):
        enhanced = F.relu(self.conv1(features.unsqueeze(1)))
        return self.conv2(enhanced).squeeze(1)


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
        return self.classifier(sequence[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_projection = nn.Linear(1, 64)
        self.position_embedding = nn.Parameter(torch.empty(1, 25, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.final_norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, NUM_CLASSES)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, features):
        tokens = self.scalar_projection(features.unsqueeze(-1))
        tokens = self.encoder(tokens + self.position_embedding)
        return self.classifier(self.final_norm(tokens.mean(dim=1)))


def make_model(family):
    if family == "gru":
        return GRUClassifier()
    if family == "transformer":
        return TransformerClassifier()
    raise ValueError(f"Unsupported model family: {family}")


def trainable_parameters(module):
    return int(sum(parameter.numel() for parameter in module.parameters()))


def make_initial_states(config):
    expected = {"gru": 40034, "transformer": 71010}
    states = {}
    hashes = {}
    metadata = {}
    for family in sorted(set(config["client_model_families"])):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["initialization_seed"])
            model = make_model(family)
        count = trainable_parameters(model)
        if count != expected[family]:
            raise RuntimeError(
                f"{family} parameter count {count} does not match {expected[family]}"
            )
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, len(FEATURE_COLUMNS), dtype=torch.float32))
        if output.shape != (2, NUM_CLASSES):
            raise RuntimeError(f"{family} output shape is {tuple(output.shape)}")
        state = cpu_state_dict(model)
        states[family] = state
        hashes[family] = state_hash(state)
        metadata[family] = {"trainable_parameters": count, "output_shape": [None, 34]}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["initialization_seed"])
        proxy = ProxyFeatureExtractor()
    proxy_state = cpu_state_dict(proxy)
    proxy_count = trainable_parameters(proxy)
    if proxy_count != 57:
        raise RuntimeError(f"Proxy parameter count {proxy_count} does not match 57")
    proxy.eval()
    with torch.inference_mode():
        proxy_output = proxy(torch.zeros(2, len(FEATURE_COLUMNS), dtype=torch.float32))
    if proxy_output.shape != (2, len(FEATURE_COLUMNS)):
        raise RuntimeError(f"Proxy output shape is {tuple(proxy_output.shape)}")
    metadata["proxy"] = {
        "trainable_parameters": proxy_count,
        "input_shape": [None, 25],
        "output_shape": [None, 25],
    }
    return states, hashes, metadata, proxy_state, state_hash(proxy_state)


def metrics_from_confusion(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    total = int(matrix.sum())
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    true_negative = float(total) - true_positive - false_positive - false_negative
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
    fpr = np.divide(
        false_positive,
        false_positive + true_negative,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(false_positive + true_negative) > 0,
    )
    fnr = np.divide(
        false_negative,
        false_negative + true_positive,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(false_negative + true_positive) > 0,
    )
    weights = support / support.sum() if support.sum() else np.zeros(NUM_CLASSES)
    benign_index = 1
    benign_true = support[benign_index]
    benign_predicted = predicted[benign_index]
    benign_correct = true_positive[benign_index]
    binary_tn = benign_correct
    binary_fp = benign_true - benign_correct
    binary_fn = benign_predicted - benign_correct
    binary_tp = float(total) - binary_tn - binary_fn - binary_fp
    binary_precision = binary_tp / (binary_tp + binary_fp) if binary_tp + binary_fp else 0.0
    binary_recall = binary_tp / (binary_tp + binary_fn) if binary_tp + binary_fn else 0.0
    binary_f1 = (
        2.0 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )
    binary_fpr = binary_fp / (binary_fp + binary_tn) if binary_fp + binary_tn else 0.0
    binary_fnr = binary_fn / (binary_fn + binary_tp) if binary_fn + binary_tp else 0.0
    return {
        "examples": total,
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "multiclass_macro_fpr": float(fpr.mean()),
        "multiclass_macro_fnr": float(fnr.mean()),
        "binary_attack_precision": float(binary_precision),
        "binary_attack_recall": float(binary_recall),
        "binary_attack_f1": float(binary_f1),
        "binary_attack_fpr": float(binary_fpr),
        "binary_attack_fnr": float(binary_fnr),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_fpr": fpr.tolist(),
        "per_class_fnr": fnr.tolist(),
        "per_class_support": support.astype(np.int64).tolist(),
    }


def csv_data_rows(path):
    newline_count = 0
    last_byte = b""
    with open(path, "rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    physical_lines = newline_count if last_byte == b"\n" else newline_count + 1
    return max(0, physical_lines - 1)


def validate_csv_header(path):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {columns}")


def validate_label_mapping(path):
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["Encoded_ID", "Label_Name"]:
        raise ValueError("label_mapping.csv has an invalid header")
    ids = frame["Encoded_ID"].to_numpy()
    if not np.array_equal(ids, np.arange(NUM_CLASSES)):
        raise ValueError("label_mapping.csv IDs must be exactly 0..33")
    names = frame["Label_Name"].astype(str).tolist()
    if names[1] != "BENIGN" or len(set(names)) != NUM_CLASSES:
        raise ValueError("label_mapping.csv names are invalid")
    return {int(index): name for index, name in zip(ids, names)}


def prepare_csv_cache(csv_path, cache_prefix, chunk_rows, split_seed, client_id=None):
    csv_path = Path(csv_path)
    validate_csv_header(csv_path)
    metadata_path = Path(str(cache_prefix) + "_metadata.json")
    if metadata_path.is_file():
        cached_result = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            validate_numpy_classification_cache(
                cached_result, require_split=client_id is not None
            )
            return cached_result
        except Exception:
            pass
    row_count = csv_data_rows(csv_path)
    if row_count <= 0:
        raise ValueError(f"No rows in {csv_path}")
    feature_path = Path(str(cache_prefix) + "_features.npy")
    label_path = Path(str(cache_prefix) + "_labels.npy")
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(row_count, len(FEATURE_COLUMNS))
    )
    labels = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.int64, shape=(row_count,)
    )
    class_chunks = [[] for _ in range(NUM_CLASSES)] if client_id is not None else None
    cursor = 0
    dtype_map = {column: np.float32 for column in FEATURE_COLUMNS}
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows, dtype=dtype_map):
        if chunk.columns.tolist() != EXPECTED_COLUMNS:
            raise ValueError(f"Header changed while reading {csv_path}")
        x = chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        raw_labels = pd.to_numeric(chunk[LABEL_COLUMN], errors="raise").to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(raw_labels).all():
            raise ValueError(f"NaN or Inf found in {csv_path}")
        if not np.equal(raw_labels, np.floor(raw_labels)).all():
            raise ValueError(f"Non-integral labels found in {csv_path}")
        y = raw_labels.astype(np.int64, copy=False)
        if y.min(initial=0) < 0 or y.max(initial=0) >= NUM_CLASSES:
            raise ValueError(f"Labels outside 0..33 found in {csv_path}")
        end = cursor + len(chunk)
        if end > row_count:
            raise ValueError(f"Row count overflow for {csv_path}")
        features[cursor:end] = x
        labels[cursor:end] = y
        if class_chunks is not None:
            for class_id in range(NUM_CLASSES):
                local = np.flatnonzero(y == class_id)
                if local.size:
                    class_chunks[class_id].append(local.astype(np.int64) + cursor)
        cursor = end
    features.flush()
    labels.flush()
    if cursor != row_count:
        raise ValueError(f"Expected {row_count} rows but parsed {cursor} from {csv_path}")
    class_counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(np.int64)
    result = {
        "features": str(feature_path),
        "labels": str(label_path),
        "rows": int(row_count),
        "class_counts": class_counts.tolist(),
    }
    if client_id is not None:
        train_parts = []
        validation_parts = []
        for class_id in range(NUM_CLASSES):
            if class_chunks[class_id]:
                indices = np.concatenate(class_chunks[class_id])
            else:
                indices = np.empty(0, dtype=np.int64)
            rng = np.random.default_rng(
                derive_seed(split_seed, "client_split", client_id, class_id)
            )
            rng.shuffle(indices)
            if indices.size <= 1:
                validation_count = 0
            else:
                validation_count = min(
                    indices.size - 1,
                    max(1, int(round(indices.size * 0.05))),
                )
            validation_parts.append(indices[:validation_count])
            train_parts.append(indices[validation_count:])
        train_indices = np.concatenate(train_parts)
        validation_indices = np.concatenate(validation_parts)
        np.random.default_rng(
            derive_seed(split_seed, "client_split_shuffle", client_id)
        ).shuffle(train_indices)
        np.random.default_rng(
            derive_seed(split_seed, "client_validation_shuffle", client_id)
        ).shuffle(validation_indices)
        train_path = Path(str(cache_prefix) + "_train_idx.npy")
        validation_path = Path(str(cache_prefix) + "_validation_idx.npy")
        train_map = np.lib.format.open_memmap(
            train_path, mode="w+", dtype=np.int64, shape=train_indices.shape
        )
        validation_map = np.lib.format.open_memmap(
            validation_path,
            mode="w+",
            dtype=np.int64,
            shape=validation_indices.shape,
        )
        train_map[:] = train_indices
        validation_map[:] = validation_indices
        train_map.flush()
        validation_map.flush()
        result.update(
            {
                "train_idx": str(train_path),
                "validation_idx": str(validation_path),
                "train_rows": int(train_indices.size),
                "validation_rows": int(validation_indices.size),
            }
        )
    write_json(metadata_path, result)
    return result


def validate_numpy_classification_cache(paths, require_split=True):
    features = np.load(paths["features"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("Cached features must have shape [N,25]")
    if features.dtype != np.float32 or labels.dtype != np.int64:
        raise ValueError("Cached features/labels must be float32/int64")
    if labels.shape != (features.shape[0],):
        raise ValueError("Cached feature/label row counts differ")
    for start in range(0, features.shape[0], 1_000_000):
        if not np.isfinite(features[start : start + 1_000_000]).all():
            raise ValueError("Cached features contain NaN or Inf")
    if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError("Cached labels are outside 0..33")
    if require_split:
        train_indices = np.load(paths["train_idx"], mmap_mode="r")
        validation_indices = np.load(paths["validation_idx"], mmap_mode="r")
        for name, indices in [
            ("train", train_indices),
            ("validation", validation_indices),
        ]:
            if indices.dtype != np.int64 or indices.ndim != 1:
                raise ValueError(f"{name} indices must be one-dimensional int64")
            if indices.size and (indices.min() < 0 or indices.max() >= features.shape[0]):
                raise ValueError(f"{name} indices are out of range")
        if train_indices.size + validation_indices.size != features.shape[0]:
            raise ValueError("Train/validation split does not account for every row")
        seen = np.zeros(features.shape[0], dtype=np.bool_)
        for indices in [train_indices, validation_indices]:
            for start in range(0, indices.size, 1_000_000):
                block = np.asarray(indices[start : start + 1_000_000])
                if np.unique(block).size != block.size or seen[block].any():
                    raise ValueError("Train/validation indices contain duplicates")
                seen[block] = True
        if not seen.all():
            raise ValueError("Train/validation indices do not cover every row")
    elif "all_idx" in paths:
        all_indices = np.load(paths["all_idx"], mmap_mode="r")
        if all_indices.dtype != np.int64 or all_indices.shape != (features.shape[0],):
            raise ValueError("Test indices must account for every row as int64")
        if all_indices.size and (
            all_indices.min() < 0 or all_indices.max() >= features.shape[0]
        ):
            raise ValueError("Test indices are out of range")
        for start in range(0, all_indices.size, 1_000_000):
            expected = np.arange(start, min(all_indices.size, start + 1_000_000))
            if not np.array_equal(all_indices[start : start + 1_000_000], expected):
                raise ValueError("Test indices must be exactly 0..N-1")
    return True


class MemmapDataset(Dataset):
    def __init__(self, paths):
        self.features = np.load(paths["features"], mmap_mode="r")
        self.labels = np.load(paths["labels"], mmap_mode="r")

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


def make_cpu_loader(paths, indices_key, batch_size, seed=None):
    dataset = MemmapDataset(paths)
    indices = np.load(paths[indices_key], mmap_mode="r")
    if seed is None:
        return DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampler = SubsetRandomSampler(indices, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def array_nbytes(path):
    array = np.load(path, mmap_mode="r")
    return int(array.size * array.dtype.itemsize)


class GPUClientCache:
    def __init__(self, device, budget_bytes, chunk_rows, logger):
        self.device = device
        self.budget_bytes = int(budget_bytes)
        self.chunk_rows = int(chunk_rows)
        self.logger = logger
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.fallbacks = 0

    def required_bytes(self, paths, require_split=True):
        keys = ["features", "labels"]
        if require_split:
            keys.extend(["train_idx", "validation_idx"])
        elif "all_idx" in paths:
            keys.append("all_idx")
        return int(sum(array_nbytes(paths[key]) for key in keys))

    def _copy_array(self, path):
        source = np.load(path, mmap_mode="r")
        dtype_map = {np.dtype(np.float32): torch.float32, np.dtype(np.int64): torch.int64}
        if source.dtype not in dtype_map:
            raise ValueError(f"Unsupported cache dtype: {source.dtype}")
        target = torch.empty(source.shape, dtype=dtype_map[source.dtype], device=self.device)
        for start in range(0, source.shape[0], self.chunk_rows):
            end = min(source.shape[0], start + self.chunk_rows)
            cpu_chunk = torch.from_numpy(np.array(source[start:end], copy=True))
            target[start:end].copy_(cpu_chunk, non_blocking=False)
        return target

    def _evict_until(self, required):
        while self.entries and self.used_bytes + required > self.budget_bytes:
            _, entry = self.entries.popitem(last=False)
            self.used_bytes -= entry["bytes"]
            self.evictions += 1
            del entry
            torch.cuda.empty_cache()

    def get(self, cache_id, paths, require_split=True):
        if cache_id in self.entries:
            self.hits += 1
            entry = self.entries.pop(cache_id)
            self.entries[cache_id] = entry
            return entry
        self.misses += 1
        validate_numpy_classification_cache(paths, require_split=require_split)
        required = self.required_bytes(paths, require_split=require_split)
        self._evict_until(required)
        if required > self.budget_bytes:
            self.fallbacks += 1
            return {"mode": "pinned_cpu_lru_fallback", "paths": paths, "bytes": 0}
        keys = ["features", "labels"]
        if require_split:
            keys.extend(["train_idx", "validation_idx"])
        elif "all_idx" in paths:
            keys.append("all_idx")
        entry = {"mode": "gpu_full_lru", "paths": paths, "bytes": required}
        for key in keys:
            entry[key] = self._copy_array(paths[key])
        self.entries[cache_id] = entry
        self.used_bytes += required
        return entry

    def snapshot(self):
        return {
            "cache_budget_bytes": self.budget_bytes,
            "cache_bytes": self.used_bytes,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "cache_fallbacks": self.fallbacks,
            "cached_ids": list(self.entries.keys()),
        }


def make_training_context(
    model,
    proxy_state,
    train_examples,
    config,
    device,
    round_index,
    client_id,
):
    proxy = ProxyFeatureExtractor().to(device)
    proxy.load_state_dict(proxy_state, strict=True)
    local_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    proxy_optimizer = torch.optim.SGD(
        proxy.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    local_scaler = GradScaler("cuda")
    proxy_scaler = GradScaler("cuda")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    generator = torch.Generator(device=device)
    generator.manual_seed(
        derive_seed(config["seed"], "local_model", round_index, client_id)
    )
    with torch.cuda.stream(stream):
        order = torch.randperm(train_examples, generator=generator, device=device)
        loss_sums = torch.zeros(4, dtype=torch.float64, device=device)
    return {
        "model": model,
        "proxy": proxy,
        "local_optimizer": local_optimizer,
        "proxy_optimizer": proxy_optimizer,
        "local_scaler": local_scaler,
        "proxy_scaler": proxy_scaler,
        "stream": stream,
        "order": order,
        "loss_sums": loss_sums,
    }


def train_client(model, family, proxy_state, cache, paths, config, device, round_index, client_id):
    started = time.perf_counter()
    client_seed = derive_seed(config["seed"], "client_train", round_index, client_id)
    torch.manual_seed(client_seed)
    torch.cuda.manual_seed(client_seed)
    np.random.seed(client_seed)
    entry = cache.get(f"client_{client_id}", paths, require_split=True)
    train_examples = int(paths["train_rows"])
    validation_examples = int(paths["validation_rows"])
    context = make_training_context(
        model,
        proxy_state,
        train_examples,
        config,
        device,
        round_index,
        client_id,
    )
    model = context["model"]
    proxy = context["proxy"]
    stream = context["stream"]
    order = context["order"]
    loss_sums = context["loss_sums"]
    batch_size = config["per_client_batch_size"]
    model.train()
    proxy.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in proxy.parameters():
        parameter.requires_grad_(False)
    local_started = time.perf_counter()
    local_steps = 0
    with torch.cuda.stream(stream):
        if entry["mode"] == "gpu_full_lru":
            train_indices = entry["train_idx"]
            for epoch in range(config["local_epochs"]):
                if epoch:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        derive_seed(config["seed"], "local_model", round_index, client_id, epoch)
                    )
                    order = torch.randperm(train_examples, generator=generator, device=device)
                for start in range(0, train_examples, batch_size):
                    end = min(train_examples, start + batch_size)
                    positions = order[start:end]
                    rows = train_indices.index_select(0, positions)
                    features = entry["features"].index_select(0, rows)
                    labels = entry["labels"].index_select(0, rows)
                    context["local_optimizer"].zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        with torch.no_grad():
                            enhanced = proxy(features)
                        enhanced_logits = model(enhanced)
                        original_logits = model(features)
                        enhanced_loss = F.cross_entropy(enhanced_logits, labels)
                        original_loss = F.cross_entropy(original_logits, labels)
                        total_loss = (
                            config["mu"] * enhanced_loss
                            + (1.0 - config["mu"]) * original_loss
                        )
                    context["local_scaler"].scale(total_loss).backward()
                    context["local_scaler"].step(context["local_optimizer"])
                    context["local_scaler"].update()
                    loss_sums[:3].add_(
                        torch.stack([enhanced_loss, original_loss, total_loss])
                        .detach()
                        .to(torch.float64)
                        * (end - start)
                    )
                    loss_sums[3].add_(end - start)
                    local_steps += 1
        else:
            for epoch in range(config["local_epochs"]):
                loader = make_cpu_loader(
                    paths,
                    "train_idx",
                    batch_size,
                    derive_seed(config["seed"], "local_cpu", round_index, client_id, epoch),
                )
                for cpu_features, cpu_labels in loader:
                    features = cpu_features.to(device, non_blocking=True)
                    labels = cpu_labels.to(device, non_blocking=True)
                    context["local_optimizer"].zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        with torch.no_grad():
                            enhanced = proxy(features)
                        enhanced_logits = model(enhanced)
                        original_logits = model(features)
                        enhanced_loss = F.cross_entropy(enhanced_logits, labels)
                        original_loss = F.cross_entropy(original_logits, labels)
                        total_loss = (
                            config["mu"] * enhanced_loss
                            + (1.0 - config["mu"]) * original_loss
                        )
                    context["local_scaler"].scale(total_loss).backward()
                    context["local_scaler"].step(context["local_optimizer"])
                    context["local_scaler"].update()
                    batch_examples = int(cpu_labels.shape[0])
                    loss_sums[:3].add_(
                        torch.stack([enhanced_loss, original_loss, total_loss])
                        .detach()
                        .to(torch.float64)
                        * batch_examples
                    )
                    loss_sums[3].add_(batch_examples)
                    local_steps += 1
    stream.synchronize()
    local_seconds = time.perf_counter() - local_started
    context["local_optimizer"].zero_grad(set_to_none=True)
    model.train()
    proxy.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in proxy.parameters():
        parameter.requires_grad_(True)
    proxy_phase_seed = derive_seed(
        config["seed"], "proxy_model", round_index, client_id
    )
    torch.manual_seed(proxy_phase_seed)
    torch.cuda.manual_seed(proxy_phase_seed)
    proxy_started = time.perf_counter()
    proxy_loss_sums = None
    proxy_steps = 0
    with torch.cuda.stream(stream):
        proxy_generator = torch.Generator(device=device)
        proxy_generator.manual_seed(
            derive_seed(config["seed"], "proxy", round_index, client_id)
        )
        proxy_order = torch.randperm(
            train_examples, generator=proxy_generator, device=device
        )
        proxy_loss_sums = torch.zeros(2, dtype=torch.float64, device=device)
        if entry["mode"] == "gpu_full_lru":
            train_indices = entry["train_idx"]
            for epoch in range(config["proxy_epochs"]):
                if epoch:
                    proxy_generator = torch.Generator(device=device)
                    proxy_generator.manual_seed(
                        derive_seed(config["seed"], "proxy", round_index, client_id, epoch)
                    )
                    proxy_order = torch.randperm(
                        train_examples, generator=proxy_generator, device=device
                    )
                for start in range(0, train_examples, batch_size):
                    end = min(train_examples, start + batch_size)
                    positions = proxy_order[start:end]
                    rows = train_indices.index_select(0, positions)
                    features = entry["features"].index_select(0, rows)
                    labels = entry["labels"].index_select(0, rows)
                    context["proxy_optimizer"].zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        logits = model(proxy(features))
                        proxy_loss = F.cross_entropy(logits, labels)
                    context["proxy_scaler"].scale(proxy_loss).backward()
                    context["proxy_scaler"].step(context["proxy_optimizer"])
                    context["proxy_scaler"].update()
                    proxy_loss_sums[0].add_(proxy_loss.detach().to(torch.float64) * (end - start))
                    proxy_loss_sums[1].add_(end - start)
                    proxy_steps += 1
        else:
            for epoch in range(config["proxy_epochs"]):
                loader = make_cpu_loader(
                    paths,
                    "train_idx",
                    batch_size,
                    derive_seed(config["seed"], "proxy_cpu", round_index, client_id, epoch),
                )
                for cpu_features, cpu_labels in loader:
                    features = cpu_features.to(device, non_blocking=True)
                    labels = cpu_labels.to(device, non_blocking=True)
                    context["proxy_optimizer"].zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        logits = model(proxy(features))
                        proxy_loss = F.cross_entropy(logits, labels)
                    context["proxy_scaler"].scale(proxy_loss).backward()
                    context["proxy_scaler"].step(context["proxy_optimizer"])
                    context["proxy_scaler"].update()
                    batch_examples = int(cpu_labels.shape[0])
                    proxy_loss_sums[0].add_(
                        proxy_loss.detach().to(torch.float64) * batch_examples
                    )
                    proxy_loss_sums[1].add_(batch_examples)
                    proxy_steps += 1
    stream.synchronize()
    proxy_seconds = time.perf_counter() - proxy_started
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    local_values = loss_sums.detach().cpu().tolist()
    proxy_values = proxy_loss_sums.detach().cpu().tolist()
    expected_local_examples = train_examples * config["local_epochs"]
    expected_proxy_examples = train_examples * config["proxy_epochs"]
    if int(local_values[3]) != expected_local_examples:
        raise RuntimeError("Local-model sample accounting is incomplete")
    if int(proxy_values[1]) != expected_proxy_examples:
        raise RuntimeError("Proxy sample accounting is incomplete")
    if local_steps != math.ceil(train_examples / batch_size) * config["local_epochs"]:
        raise RuntimeError("Local-model optimizer-step accounting is incomplete")
    if proxy_steps != math.ceil(train_examples / batch_size) * config["proxy_epochs"]:
        raise RuntimeError("Proxy optimizer-step accounting is incomplete")
    validation_started = time.perf_counter()
    validation_loss_sum, validation_confusion = evaluate_model(
        model,
        entry,
        paths,
        "validation_idx",
        validation_examples,
        batch_size,
        device,
    )
    validation_seconds = time.perf_counter() - validation_started
    validation_metrics = metrics_from_confusion(validation_confusion)
    if int(validation_confusion.sum()) != validation_examples:
        raise RuntimeError("Local validation sample accounting is incomplete")
    validation_metrics["loss"] = (
        validation_loss_sum / validation_examples if validation_examples else 0.0
    )
    denominator = max(1.0, local_values[3])
    proxy_denominator = max(1.0, proxy_values[1])
    result = {
        "client_id": client_id,
        "model_family": family,
        "round": round_index,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "enhanced_cross_entropy_loss": local_values[0] / denominator,
        "original_cross_entropy_loss": local_values[1] / denominator,
        "local_model_total_loss": local_values[2] / denominator,
        "proxy_cross_entropy_loss": proxy_values[0] / proxy_denominator,
        "local_optimizer_steps": local_steps,
        "proxy_optimizer_steps": proxy_steps,
        "local_model_seconds": local_seconds,
        "proxy_seconds": proxy_seconds,
        "validation_seconds": validation_seconds,
        "actual_client_seconds": time.perf_counter() - started,
        "validation_metrics": validation_metrics,
        "validation_confusion": validation_confusion,
        "personalized_state": cpu_state_dict(model),
        "local_proxy_state": cpu_state_dict(proxy),
        "gpu_cache_mode": entry["mode"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    return result


def evaluate_model(model, entry, paths, indices_key, examples, batch_size, device):
    model.eval()
    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    with torch.inference_mode():
        if entry["mode"] == "gpu_full_lru":
            indices = entry[indices_key]
            for start in range(0, examples, batch_size):
                end = min(examples, start + batch_size)
                rows = indices[start:end]
                features = entry["features"].index_select(0, rows)
                labels = entry["labels"].index_select(0, rows)
                with autocast("cuda"):
                    logits = model(features)
                    loss = F.cross_entropy(logits, labels)
                predictions = logits.argmax(dim=1)
                bins = torch.bincount(
                    labels * NUM_CLASSES + predictions,
                    minlength=NUM_CLASSES * NUM_CLASSES,
                ).reshape(NUM_CLASSES, NUM_CLASSES)
                confusion.add_(bins)
                loss_sum.add_(loss.detach().to(torch.float64) * (end - start))
        else:
            loader = make_cpu_loader(paths, indices_key, batch_size)
            for cpu_features, cpu_labels in loader:
                features = cpu_features.to(device, non_blocking=True)
                labels = cpu_labels.to(device, non_blocking=True)
                with autocast("cuda"):
                    logits = model(features)
                    loss = F.cross_entropy(logits, labels)
                predictions = logits.argmax(dim=1)
                bins = torch.bincount(
                    labels * NUM_CLASSES + predictions,
                    minlength=NUM_CLASSES * NUM_CLASSES,
                ).reshape(NUM_CLASSES, NUM_CLASSES)
                confusion.add_(bins)
                loss_sum.add_(loss.detach().to(torch.float64) * int(cpu_labels.shape[0]))
    torch.cuda.synchronize(device)
    return float(loss_sum.cpu()), confusion.cpu().numpy()


def benchmark_family(family, config, device):
    seed_everything(derive_seed(config["seed"], "benchmark", family, device.index))
    model = make_model(family).to(device).train()
    proxy = ProxyFeatureExtractor().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    proxy_optimizer = torch.optim.SGD(proxy.parameters(), lr=config["learning_rate"])
    scaler = GradScaler("cuda")
    proxy_scaler = GradScaler("cuda")
    batch_size = config["per_client_batch_size"]
    features = torch.randn(batch_size, 25, device=device)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    for warmup in range(2):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            enhanced = proxy(features).detach()
            local_loss = (
                config["mu"] * F.cross_entropy(model(enhanced), labels)
                + (1.0 - config["mu"]) * F.cross_entropy(model(features), labels)
            )
        scaler.scale(local_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize(device)
    local_start = torch.cuda.Event(enable_timing=True)
    local_end = torch.cuda.Event(enable_timing=True)
    local_start.record()
    for measured in range(3):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            enhanced = proxy(features).detach()
            local_loss = (
                config["mu"] * F.cross_entropy(model(enhanced), labels)
                + (1.0 - config["mu"]) * F.cross_entropy(model(features), labels)
            )
        scaler.scale(local_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    local_end.record()
    torch.cuda.synchronize(device)
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    proxy_start = torch.cuda.Event(enable_timing=True)
    proxy_end = torch.cuda.Event(enable_timing=True)
    proxy_start.record()
    for measured in range(3):
        proxy_optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            proxy_loss = F.cross_entropy(model(proxy(features)), labels)
        proxy_scaler.scale(proxy_loss).backward()
        proxy_scaler.step(proxy_optimizer)
        proxy_scaler.update()
    proxy_end.record()
    torch.cuda.synchronize(device)
    return {
        "local_step_seconds": local_start.elapsed_time(local_end) / 3000.0,
        "proxy_step_seconds": proxy_start.elapsed_time(proxy_end) / 3000.0,
    }


def benchmark_stream_candidates(families, config, device):
    candidates = config["stream_candidates"]
    results = {}
    for stream_count in candidates:
        active_families = [families[index % len(families)] for index in range(stream_count)]
        streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
        models = [make_model(family).to(device).train() for family in active_families]
        proxies = [ProxyFeatureExtractor().to(device).eval() for _ in active_families]
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
            for model in models
        ]
        scalers = [GradScaler("cuda") for _ in active_families]
        feature_batches = [
            torch.randn(config["per_client_batch_size"], 25, device=device)
            for _ in active_families
        ]
        label_batches = [
            torch.randint(
                0,
                NUM_CLASSES,
                (config["per_client_batch_size"],),
                device=device,
            )
            for _ in active_families
        ]
        default_stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(default_stream)
        for stream in streams:
            stream.wait_event(start)
        for measured in range(3):
            for index in range(stream_count):
                with torch.cuda.stream(streams[index]):
                    optimizers[index].zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        enhanced = proxies[index](feature_batches[index]).detach()
                        loss = F.cross_entropy(models[index](enhanced), label_batches[index])
                    scalers[index].scale(loss).backward()
                    scalers[index].step(optimizers[index])
                    scalers[index].update()
        for stream in streams:
            default_stream.wait_stream(stream)
        end.record(default_stream)
        torch.cuda.synchronize(device)
        seconds = start.elapsed_time(end) / 1000.0
        samples = 3 * stream_count * config["per_client_batch_size"]
        results[str(stream_count)] = {
            "seconds": seconds,
            "samples_per_second": samples / seconds,
        }
        del models, proxies, optimizers, scalers, feature_batches, label_batches
        torch.cuda.empty_cache()
    best_measured = max(
        candidates,
        key=lambda value: results[str(value)]["samples_per_second"],
    )
    selected = 1 if config["stochastic_models_force_single_stream"] else best_measured
    return {
        "candidates": results,
        "best_measured": best_measured,
        "selected": selected,
        "selection_reason": (
            "single stream preserves dropout RNG reproducibility"
            if config["stochastic_models_force_single_stream"]
            else "highest measured aggregate throughput"
        ),
    }


def worker_main(gpu_id, task_queue, result_queue, manifest_path):
    worker_logger = None
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        config = manifest["config"]
        output_dir = Path(config["output_dir"])
        log_path = (
            output_dir / "logs" / "rank_1.log"
            if gpu_id == 1
            else Path(config["temp_dir"]) / "worker_0.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        worker_logger = configure_logger(log_path, f"pfedes_worker_{gpu_id}")
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
        if "T4" not in torch.cuda.get_device_name(device):
            raise RuntimeError(f"GPU {gpu_id} is not an NVIDIA T4")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        properties = torch.cuda.get_device_properties(device)
        persistent_model_reserve_bytes = 64 * 1024 * 1024
        cache_budget_bytes = max(
            0,
            int(properties.total_memory * config["gpu_cache_fraction"])
            - persistent_model_reserve_bytes,
        )
        cache = GPUClientCache(
            device,
            cache_budget_bytes,
            config["gpu_copy_chunk_rows"],
            worker_logger,
        )
        models = {}
        assigned_clients = []
        client_paths = {}
        selected_streams = 1
        while True:
            task = task_queue.get()
            kind = task["kind"]
            if kind == "shutdown":
                break
            if kind == "benchmark":
                benchmarks = {
                    family: benchmark_family(family, config, device)
                    for family in task["families"]
                }
                result_queue.put(
                    {"kind": "benchmark_complete", "gpu_id": gpu_id, "benchmarks": benchmarks}
                )
                continue
            if kind == "initialize":
                initialize_started = time.perf_counter()
                assigned_clients = task["assigned_clients"]
                client_paths = task["client_paths"]
                for client_id in assigned_clients:
                    family = config["client_model_families"][client_id - 1]
                    model = make_model(family).to(device)
                    model.load_state_dict(task["personalized_states"][client_id], strict=True)
                    models[client_id] = model
                    cache.get(f"client_{client_id}", client_paths[str(client_id)], require_split=True)
                family_list = sorted(
                    set(config["client_model_families"][client_id - 1] for client_id in assigned_clients)
                )
                stream_benchmark = benchmark_stream_candidates(family_list, config, device)
                selected_streams = stream_benchmark["selected"]
                result_queue.put(
                    {
                        "kind": "initialize_complete",
                        "gpu_id": gpu_id,
                        "assigned_clients": assigned_clients,
                        "cache": cache.snapshot(),
                        "stream_benchmark": stream_benchmark,
                        "selected_streams": selected_streams,
                        "initialization_seconds": time.perf_counter() - initialize_started,
                        "persistent_model_reserve_bytes": persistent_model_reserve_bytes,
                    }
                )
                continue
            if kind == "train_round":
                torch.cuda.reset_peak_memory_stats(device)
                round_started = time.perf_counter()
                client_results = []
                for client_id in assigned_clients:
                    family = config["client_model_families"][client_id - 1]
                    client_results.append(
                        train_client(
                            models[client_id],
                            family,
                            task["global_proxy_state"],
                            cache,
                            client_paths[str(client_id)],
                            config,
                            device,
                            task["round"],
                            client_id,
                        )
                    )
                payload_path = (
                    Path(config["payload_dir"])
                    / f"gpu_{gpu_id}_round_{task['round']}.pt"
                )
                payload_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(client_results, payload_path)
                result_queue.put(
                    {
                        "kind": "round_complete",
                        "gpu_id": gpu_id,
                        "round": task["round"],
                        "payload_path": str(payload_path),
                        "worker_seconds": time.perf_counter() - round_started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                        "cache": cache.snapshot(),
                        "selected_streams": selected_streams,
                    }
                )
                continue
            if kind == "final_test":
                test_entry = cache.get("global_test", task["test_paths"], require_split=False)
                test_results = []
                for client_id in assigned_clients:
                    models[client_id].load_state_dict(
                        task["personalized_states"][client_id], strict=True
                    )
                    loss_sum, confusion = evaluate_model(
                        models[client_id],
                        test_entry,
                        task["test_paths"],
                        "all_idx",
                        int(task["test_paths"]["rows"]),
                        config["per_client_batch_size"],
                        device,
                    )
                    metrics = metrics_from_confusion(confusion)
                    metrics["loss"] = loss_sum / int(task["test_paths"]["rows"])
                    test_results.append(
                        {
                            "client_id": client_id,
                            "model_family": config["client_model_families"][client_id - 1],
                            "metrics": metrics,
                            "confusion": confusion,
                        }
                    )
                payload_path = Path(config["payload_dir"]) / f"gpu_{gpu_id}_final_test.pt"
                torch.save(test_results, payload_path)
                result_queue.put(
                    {
                        "kind": "test_complete",
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                        "cache": cache.snapshot(),
                    }
                )
                continue
            raise ValueError(f"Unknown worker task: {kind}")
        worker_logger.info("Worker %s stopped cleanly", gpu_id)
    except Exception:
        error_text = traceback.format_exc()
        if worker_logger is not None:
            worker_logger.exception("Worker %s failed", gpu_id)
        result_queue.put(
            {"kind": "worker_error", "gpu_id": gpu_id, "traceback": error_text}
        )
        raise


def collect_worker_results(result_queue, workers, expected_kind, expected_count, timeout_seconds):
    results = []
    deadline = time.monotonic() + timeout_seconds
    seen_gpu_ids = set()
    while len(results) < expected_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {expected_kind}")
        try:
            result = result_queue.get(timeout=min(30.0, remaining))
        except queue.Empty:
            failed = [worker.pid for worker in workers if not worker.is_alive()]
            if failed:
                raise RuntimeError(f"Workers exited before completing: {failed}")
            continue
        if result.get("kind") == "worker_error":
            raise RuntimeError(result["traceback"])
        if result.get("kind") != expected_kind:
            raise RuntimeError(f"Expected {expected_kind}, received {result.get('kind')}")
        gpu_id = result["gpu_id"]
        if gpu_id in seen_gpu_ids:
            raise RuntimeError(f"Duplicate result from GPU {gpu_id}")
        seen_gpu_ids.add(gpu_id)
        results.append(result)
    return sorted(results, key=lambda value: value["gpu_id"])


def choose_client_assignment(client_paths, config, benchmark_results):
    average_benchmarks = {}
    families = sorted(set(config["client_model_families"]))
    for family in families:
        average_benchmarks[family] = {
            key: float(np.mean([result["benchmarks"][family][key] for result in benchmark_results]))
            for key in ["local_step_seconds", "proxy_step_seconds"]
        }
    workloads = {}
    for client_id in range(1, 11):
        family = config["client_model_families"][client_id - 1]
        steps = math.ceil(
            int(client_paths[str(client_id)]["train_rows"])
            / config["per_client_batch_size"]
        )
        rates = average_benchmarks[family]
        workloads[client_id] = steps * (
            config["local_epochs"] * rates["local_step_seconds"]
            + config["proxy_epochs"] * rates["proxy_step_seconds"]
        )
    best = None
    clients = tuple(range(1, 11))
    for mask in range(1, 2**10 - 1):
        left = tuple(client for index, client in enumerate(clients) if mask & (1 << index))
        right = tuple(client for client in clients if client not in left)
        if left > right:
            continue
        load_left = sum(workloads[client] for client in left)
        load_right = sum(workloads[client] for client in right)
        objective = (max(load_left, load_right), abs(load_left - load_right), left)
        if best is None or objective < best[0]:
            best = (objective, left, right, load_left, load_right)
    if best is None:
        raise RuntimeError("Could not find a nontrivial client bipartition")
    return {
        "0": list(best[1]),
        "1": list(best[2]),
        "predicted_load_seconds": {"0": best[3], "1": best[4]},
        "client_workloads_seconds": workloads,
        "benchmarks": average_benchmarks,
    }


def weighted_average_states(states, weights):
    if not states or len(states) != len(weights):
        raise ValueError("State aggregation received incomplete inputs")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum()
    result = OrderedDict()
    for key in states[0]:
        reference = states[0][key]
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, weight in zip(states, normalized):
                accumulator.add_(state[key].to(torch.float64), alpha=float(weight))
            result[key] = accumulator.to(reference.dtype)
        else:
            result[key] = reference.clone()
    return result


def nvidia_smi_environment():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row for row in csv.reader(completed.stdout.strip().splitlines()) if row]
    if len(rows) != 2:
        raise RuntimeError(f"Expected exactly two GPUs, found {len(rows)}")
    environment = []
    for row in rows:
        index, name, memory_mib, compute_capability = [value.strip() for value in row]
        if "T4" not in name:
            raise RuntimeError(f"GPU {index} is not an NVIDIA T4: {name}")
        environment.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "compute_capability": compute_capability,
            }
        )
    return environment


class GPUSampler:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self):
        while not self.stop_event.wait(5.0):
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                timestamp = time.time()
                for row in csv.reader(completed.stdout.strip().splitlines()):
                    gpu_id, utilization, memory_used = [value.strip() for value in row]
                    self.samples.append(
                        {
                            "timestamp": timestamp,
                            "gpu_id": int(gpu_id),
                            "utilization_percent": float(utilization),
                            "memory_used_mib": float(memory_used),
                        }
                    )
            except Exception:
                continue


def prepare_data(config, logger):
    started = time.perf_counter()
    input_dir = Path(config["input_dir"])
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        *[input_dir / f"client_{client_id}_train.csv" for client_id in range(1, 11)],
        input_dir / "global_train_data.csv",
        input_dir / "global_test_data.csv",
        input_dir / "label_mapping.csv",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Kaggle inputs: {missing}")
    label_mapping = validate_label_mapping(input_dir / "label_mapping.csv")
    validate_csv_header(input_dir / "global_train_data.csv")
    global_train_rows = csv_data_rows(input_dir / "global_train_data.csv")
    client_paths = {}
    class_distribution = []
    for client_id in range(1, 11):
        logger.info("Preparing client %s memmaps", client_id)
        paths = prepare_csv_cache(
            input_dir / f"client_{client_id}_train.csv",
            cache_dir / f"client_{client_id}",
            config["csv_chunk_rows"],
            config["split_seed"],
            client_id=client_id,
        )
        validate_numpy_classification_cache(paths, require_split=True)
        client_paths[str(client_id)] = paths
        for class_id, count in enumerate(paths["class_counts"]):
            class_distribution.append(
                {
                    "client_id": client_id,
                    "class_id": class_id,
                    "class_name": label_mapping[class_id],
                    "examples": count,
                }
            )
    total_client_rows = sum(int(paths["rows"]) for paths in client_paths.values())
    logger.info("Preparing global test memmap")
    test_paths = prepare_csv_cache(
        input_dir / "global_test_data.csv",
        cache_dir / "global_test",
        config["csv_chunk_rows"],
        config["split_seed"],
        client_id=None,
    )
    all_idx_path = cache_dir / "global_test_all_idx.npy"
    all_indices = np.lib.format.open_memmap(
        all_idx_path,
        mode="w+",
        dtype=np.int64,
        shape=(int(test_paths["rows"]),),
    )
    all_indices[:] = np.arange(int(test_paths["rows"]), dtype=np.int64)
    all_indices.flush()
    test_paths["all_idx"] = str(all_idx_path)
    validate_numpy_classification_cache(test_paths, require_split=False)
    dataset_summary = {
        "input_dir": str(input_dir),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "num_classes": NUM_CLASSES,
        "label_mapping": label_mapping,
        "global_train_rows_validated_not_used": global_train_rows,
        "total_client_rows": total_client_rows,
        "client_union_matches_global_train_row_count": total_client_rows == global_train_rows,
        "global_test_rows": int(test_paths["rows"]),
        "client_rows": {key: int(value["rows"]) for key, value in client_paths.items()},
        "client_train_rows": {
            key: int(value["train_rows"]) for key, value in client_paths.items()
        },
        "client_validation_rows": {
            key: int(value["validation_rows"]) for key, value in client_paths.items()
        },
        "split_policy": "per-client stratified 95/5; singleton classes remain in train",
        "preprocessing": "already QuantileTransformer-normalized upstream; no refit",
        "data_preparation_seconds": time.perf_counter() - started,
    }
    return client_paths, test_paths, label_mapping, class_distribution, dataset_summary


def classification_report_from_metrics(metrics, label_mapping):
    report = {}
    rows = []
    for class_id in range(NUM_CLASSES):
        entry = {
            "class_id": class_id,
            "class_name": label_mapping[class_id],
            "precision": metrics["per_class_precision"][class_id],
            "recall": metrics["per_class_recall"][class_id],
            "f1": metrics["per_class_f1"][class_id],
            "support": metrics["per_class_support"][class_id],
        }
        report[label_mapping[class_id]] = entry
        rows.append(entry)
    for aggregate in ["macro", "weighted"]:
        entry = {
            "class_id": "",
            "class_name": f"{aggregate}_avg",
            "precision": metrics[f"{aggregate}_precision"],
            "recall": metrics[f"{aggregate}_recall"],
            "f1": metrics[f"{aggregate}_f1"],
            "support": metrics["examples"],
        }
        report[f"{aggregate}_avg"] = entry
        rows.append(entry)
    report["accuracy"] = metrics["accuracy"]
    return report, rows


def create_plots(output_dir, class_distribution, history_round, final_confusion, final_metrics, label_mapping):
    artifact_dir = Path(output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    distribution = np.zeros((10, NUM_CLASSES), dtype=np.int64)
    for row in class_distribution:
        distribution[row["client_id"] - 1, row["class_id"]] = row["examples"]
    fig, axis = plt.subplots(figsize=(15, 6))
    image = axis.imshow(np.log1p(distribution), aspect="auto", cmap="viridis")
    axis.set_xlabel("Class ID")
    axis.set_ylabel("Client")
    axis.set_yticks(range(10), labels=range(1, 11))
    fig.colorbar(image, ax=axis, label="log(1 + examples)")
    fig.tight_layout()
    fig.savefig(artifact_dir / "class_distribution.png", dpi=160)
    plt.close(fig)
    rounds = [row["round"] for row in history_round]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(rounds, [row["mean_local_validation_accuracy"] for row in history_round], label="mean accuracy")
    axis.plot(rounds, [row["mean_local_validation_macro_f1"] for row in history_round], label="mean macro-F1")
    axis.set_xlabel("Round")
    axis.set_ylabel("Metric")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_f1_curves.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(rounds, [row["mean_local_model_total_loss"] for row in history_round], label="local model")
    axis.plot(rounds, [row["mean_proxy_cross_entropy_loss"] for row in history_round], label="proxy")
    axis.set_xlabel("Round")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifact_dir / "loss_curves.png", dpi=160)
    plt.close(fig)
    normalized = np.divide(
        final_confusion,
        final_confusion.sum(axis=1, keepdims=True),
        out=np.zeros_like(final_confusion, dtype=np.float64),
        where=final_confusion.sum(axis=1, keepdims=True) > 0,
    )
    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(artifact_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.bar(range(NUM_CLASSES), final_metrics["per_class_f1"])
    axis.set_xticks(range(NUM_CLASSES), [label_mapping[index] for index in range(NUM_CLASSES)], rotation=90)
    axis.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(artifact_dir / "per_class_f1.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(rounds, [row["round_seconds"] for row in history_round])
    axis.set_xlabel("Round")
    axis.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(artifact_dir / "runtime_per_round.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(rounds, [row["communication_cumulative_mib"] for row in history_round], marker="o")
    axis.set_xlabel("Round")
    axis.set_ylabel("Cumulative logical communication (MiB)")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifact_dir / "communication_cumulative.png", dpi=160)
    plt.close(fig)


def verify_outputs(output_dir, config):
    for relative_path in REQUIRED_RELATIVE_OUTPUTS:
        path = Path(output_dir) / relative_path
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Required output is missing or empty: {path}")
    expected_rows = {
        "metrics/history_pretrain.csv": 1,
        "metrics/history_round.csv": config["communication_rounds"],
        "metrics/history_client.csv": config["communication_rounds"] * 10,
        "metrics/history_local_epoch.csv": config["communication_rounds"] * 10,
    }
    for relative_path, expected in expected_rows.items():
        actual = len(pd.read_csv(Path(output_dir) / relative_path))
        if actual != expected:
            raise RuntimeError(f"{relative_path} has {actual} rows; expected {expected}")


def main():
    pipeline_started = time.perf_counter()
    manifest_path = Path(os.environ["TRAINING_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if config["execution_mode"] != "client_parallel":
        raise ValueError("This runtime requires client_parallel execution")
    if config["num_clients"] != 10 or len(config["client_model_families"]) != 10:
        raise ValueError("Exactly ten client model assignments are required")
    output_dir = Path(config["output_dir"])
    for subdirectory in ["checkpoints", "logs", "metrics", "artifacts"]:
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir / "logs" / "run.log", "pfedes_coordinator")
    seed_everything(config["seed"])
    environment = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": nvidia_smi_environment(),
        "execution_mode": config["execution_mode"],
    }
    logger.info("Starting %s", config["run_name"])
    write_json(output_dir / "metrics" / "config.json", config)
    data_started = time.perf_counter()
    client_paths, test_paths, label_mapping, class_distribution, dataset_summary = prepare_data(config, logger)
    data_seconds = time.perf_counter() - data_started
    write_json(output_dir / "metrics" / "dataset_summary.json", dataset_summary)
    pd.DataFrame(class_distribution).to_csv(
        output_dir / "metrics" / "client_class_distribution.csv", index=False
    )
    initial_states, initialization_hashes, model_metadata, global_proxy_state, initial_proxy_hash = make_initial_states(config)
    initial_model_state_dicts_by_family = initial_states
    history_pretrain = [
        {
            "phase": "not_applicable_pfedes_has_no_server_pretrain",
            "enabled": False,
            "epoch": 0,
            "train_examples": 0,
            "optimizer_steps": 0,
            "cross_entropy_loss": 0.0,
            "epoch_seconds": 0.0,
            "gpu_id": -1,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    ]
    write_table_pair(
        history_pretrain,
        output_dir / "metrics" / "history_pretrain.csv",
        output_dir / "metrics" / "history_pretrain.json",
    )
    history_round = []
    history_client = []
    history_local_epoch = []
    communication_rows = []
    stream_results = []
    assignment = None
    best_score = -math.inf
    best_round = 0
    cumulative_seconds = 0.0
    cumulative_communication = 0
    personalized_states = {
        client_id: OrderedDict(
            (key, value.clone())
            for key, value in initial_states[config["client_model_families"][client_id - 1]].items()
        )
        for client_id in range(1, 11)
    }
    runtime_rounds = []
    start_round = 1
    resume_path_value = config.get("resume_checkpoint_path")
    resume_path = (
        Path(resume_path_value)
        if resume_path_value
        else output_dir / "checkpoints" / "last.pt"
    )
    if config.get("resume_if_available", True) and resume_path.is_file():
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        checkpoint_config = resume_checkpoint.get("config", {})
        if checkpoint_config.get("run_name") != config["run_name"]:
            raise ValueError("Resume checkpoint run_name does not match CONFIG")
        if checkpoint_config.get("client_model_families") != config["client_model_families"]:
            raise ValueError("Resume checkpoint model assignment does not match CONFIG")
        resume_locked_keys = [
            "local_epochs",
            "proxy_epochs",
            "per_client_batch_size",
            "learning_rate",
            "momentum",
            "weight_decay",
            "mu",
            "seed",
            "initialization_seed",
            "split_seed",
        ]
        for key in resume_locked_keys:
            if checkpoint_config.get(key) != config.get(key):
                raise ValueError(f"Resume checkpoint differs on locked key: {key}")
        completed_round = int(resume_checkpoint["round"])
        if completed_round < 1 or completed_round > config["communication_rounds"]:
            raise ValueError("Resume checkpoint round is outside the configured range")
        global_proxy_state = resume_checkpoint["global_proxy_state_dict"]
        personalized_states = resume_checkpoint["personalized_model_state_dicts"]
        history_source = resume_path.parent.parent / "metrics"
        history_round = json.loads(
            (history_source / "history_round.json").read_text(encoding="utf-8")
        )
        history_client = json.loads(
            (history_source / "history_client.json").read_text(encoding="utf-8")
        )
        history_local_epoch = json.loads(
            (history_source / "history_local_epoch.json").read_text(encoding="utf-8")
        )
        if len(history_round) < completed_round:
            raise ValueError("Resume round history is behind the checkpoint")
        if len(history_client) < completed_round * 10:
            raise ValueError("Resume client history is behind the checkpoint")
        if len(history_local_epoch) < completed_round * 10:
            raise ValueError("Resume local-epoch history is behind the checkpoint")
        history_round = history_round[:completed_round]
        history_client = history_client[: completed_round * 10]
        history_local_epoch = history_local_epoch[: completed_round * 10]
        communication_rows = [
            {
                "round": row["round"],
                "selected_clients": 10,
                "proxy_state_bytes": state_num_bytes(global_proxy_state),
                "round_bytes": row["communication_bytes"],
                "round_mib": row["communication_mib"],
                "cumulative_bytes": row["communication_cumulative_bytes"],
                "cumulative_mib": row["communication_cumulative_mib"],
            }
            for row in history_round
        ]
        runtime_rounds = [
            {
                "round": row["round"],
                "seconds": row["round_seconds"],
                "worker_seconds": {
                    "0": row["actual_worker_load_gpu0_seconds"],
                    "1": row["actual_worker_load_gpu1_seconds"],
                },
            }
            for row in history_round
        ]
        cumulative_seconds = float(history_round[-1]["cumulative_seconds"])
        cumulative_communication = int(
            history_round[-1]["communication_cumulative_bytes"]
        )
        best_source = resume_path.parent / "best.pt"
        if best_source.is_file():
            best_checkpoint = torch.load(
                best_source, map_location="cpu", weights_only=False
            )
            best_score = float(
                best_checkpoint["validation_metrics"]["mean_personalized_macro_f1"]
            )
            best_round = int(best_checkpoint["round"])
            resume_score = float(
                resume_checkpoint["validation_metrics"]["mean_personalized_macro_f1"]
            )
            if resume_score > best_score:
                best_checkpoint = resume_checkpoint
                best_score = resume_score
                best_round = completed_round
            atomic_torch_save(best_checkpoint, output_dir / "checkpoints" / "best.pt")
        else:
            raise FileNotFoundError(
                f"Resume requires the sibling best checkpoint: {best_source}"
            )
        atomic_torch_save(resume_checkpoint, output_dir / "checkpoints" / "last.pt")
        write_table_pair(
            history_round,
            output_dir / "metrics" / "history_round.csv",
            output_dir / "metrics" / "history_round.json",
        )
        write_table_pair(
            history_client,
            output_dir / "metrics" / "history_client.csv",
            output_dir / "metrics" / "history_client.json",
        )
        write_table_pair(
            history_local_epoch,
            output_dir / "metrics" / "history_local_epoch.csv",
            output_dir / "metrics" / "history_local_epoch.json",
        )
        start_round = completed_round + 1
        logger.info("Resuming after completed round %s", completed_round)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(2)]
    workers = [
        context.Process(
            target=worker_main,
            args=(gpu_id, task_queues[gpu_id], result_queue, str(manifest_path)),
            name=f"pfedes_gpu_{gpu_id}",
        )
        for gpu_id in range(2)
    ]
    for worker in workers:
        worker.start()
    sampler = GPUSampler()
    sampler.start()
    try:
        families = sorted(set(config["client_model_families"]))
        for task_queue in task_queues:
            task_queue.put({"kind": "benchmark", "families": families})
        benchmark_results = collect_worker_results(
            result_queue,
            workers,
            "benchmark_complete",
            2,
            config["worker_timeout_seconds"],
        )
        assignment = choose_client_assignment(client_paths, config, benchmark_results)
        logger.info("Client assignment: %s", assignment)
        for gpu_id, task_queue in enumerate(task_queues):
            task_queue.put(
                {
                    "kind": "initialize",
                    "assigned_clients": assignment[str(gpu_id)],
                    "client_paths": client_paths,
                    "personalized_states": personalized_states,
                }
            )
        initialization_results = collect_worker_results(
            result_queue,
            workers,
            "initialize_complete",
            2,
            config["worker_timeout_seconds"],
        )
        stream_results = [result["stream_benchmark"] for result in initialization_results]
        proxy_state_bytes = state_num_bytes(global_proxy_state)
        for round_index in range(start_round, config["communication_rounds"] + 1):
            round_started = time.perf_counter()
            for task_queue in task_queues:
                task_queue.put(
                    {
                        "kind": "train_round",
                        "round": round_index,
                        "global_proxy_state": global_proxy_state,
                    }
                )
            worker_results = collect_worker_results(
                result_queue,
                workers,
                "round_complete",
                2,
                config["worker_timeout_seconds"],
            )
            client_results = []
            for result in worker_results:
                client_results.extend(
                    torch.load(result["payload_path"], map_location="cpu", weights_only=False)
                )
            client_results.sort(key=lambda value: value["client_id"])
            if [result["client_id"] for result in client_results] != list(range(1, 11)):
                raise RuntimeError("A round did not return exactly clients 1..10")
            proxy_states = [result["local_proxy_state"] for result in client_results]
            aggregation_weights = [result["train_examples"] for result in client_results]
            global_proxy_state = weighted_average_states(proxy_states, aggregation_weights)
            pooled_validation_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for result in client_results:
                personalized_states[result["client_id"]] = result["personalized_state"]
                pooled_validation_confusion += result["validation_confusion"]
            pooled_validation_metrics = metrics_from_confusion(pooled_validation_confusion)
            mean_accuracy = float(
                np.mean([result["validation_metrics"]["accuracy"] for result in client_results])
            )
            mean_macro_f1 = float(
                np.mean([result["validation_metrics"]["macro_f1"] for result in client_results])
            )
            round_seconds = time.perf_counter() - round_started
            cumulative_seconds += round_seconds
            logical_bytes = 2 * 10 * proxy_state_bytes
            cumulative_communication += logical_bytes
            worker_times = {str(result["gpu_id"]): result["worker_seconds"] for result in worker_results}
            max_worker_time = max(worker_times.values())
            worker_idle = {
                gpu_id: max_worker_time - seconds for gpu_id, seconds in worker_times.items()
            }
            total_train_examples = sum(result["train_examples"] for result in client_results)
            mean_local_loss = sum(
                result["local_model_total_loss"] * result["train_examples"]
                for result in client_results
            ) / total_train_examples
            mean_proxy_loss = sum(
                result["proxy_cross_entropy_loss"] * result["train_examples"]
                for result in client_results
            ) / total_train_examples
            for result in client_results:
                gpu_id = 0 if result["client_id"] in assignment["0"] else 1
                metrics = result["validation_metrics"]
                history_local_epoch.append(
                    {
                        "phase": "pfedes_iterative_training",
                        "round": round_index,
                        "client_id": result["client_id"],
                        "model_family": result["model_family"],
                        "local_epoch": 1,
                        "selected_for_global_update": True,
                        "received_global_state": True,
                        "train_examples": result["train_examples"],
                        "optimizer_steps": result["local_optimizer_steps"],
                        "proxy_optimizer_steps": result["proxy_optimizer_steps"],
                        "sampler_padding_rows": 0,
                        "hard_loss": result["local_model_total_loss"],
                        "soft_loss": 0.0,
                        "proximal_loss": 0.0,
                        "total_loss": result["local_model_total_loss"],
                        "cross_entropy_loss": result["local_model_total_loss"],
                        "distillation_kl_loss": 0.0,
                        "enhanced_cross_entropy_loss": result["enhanced_cross_entropy_loss"],
                        "original_cross_entropy_loss": result["original_cross_entropy_loss"],
                        "proxy_cross_entropy_loss": result["proxy_cross_entropy_loss"],
                        "lambda": 0.0,
                        "temperature": 1.0,
                        "mu": config["mu"],
                        "epoch_seconds": result["local_model_seconds"] + result["proxy_seconds"],
                        "gpu_id": gpu_id,
                        "concurrent_streams": 1,
                        "gpu_cache_mode": result["gpu_cache_mode"],
                    }
                )
                client_row = {
                    "round": round_index,
                    "client_id": result["client_id"],
                    "model_family": result["model_family"],
                    "train_examples": result["train_examples"],
                    "validation_examples": result["validation_examples"],
                    "selected": True,
                    "received_global_proxy": True,
                    "uploaded_local_proxy": True,
                    "enhanced_cross_entropy_loss": result["enhanced_cross_entropy_loss"],
                    "original_cross_entropy_loss": result["original_cross_entropy_loss"],
                    "local_model_total_loss": result["local_model_total_loss"],
                    "proxy_cross_entropy_loss": result["proxy_cross_entropy_loss"],
                    "local_model_seconds": result["local_model_seconds"],
                    "proxy_seconds": result["proxy_seconds"],
                    "validation_seconds": result["validation_seconds"],
                    "actual_client_seconds": result["actual_client_seconds"],
                    "gpu_id": gpu_id,
                    "gpu_cache_mode": result["gpu_cache_mode"],
                    "concurrent_streams": 1,
                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                    "peak_reserved_bytes": result["peak_reserved_bytes"],
                }
                for key, value in metrics.items():
                    if not key.startswith("per_class_"):
                        client_row[f"validation_{key}"] = value
                history_client.append(client_row)
            round_row = {
                "round": round_index,
                "mean_local_model_total_loss": mean_local_loss,
                "mean_proxy_cross_entropy_loss": mean_proxy_loss,
                "mean_local_validation_accuracy": mean_accuracy,
                "mean_local_validation_macro_f1": mean_macro_f1,
                "min_local_validation_accuracy": min(result["validation_metrics"]["accuracy"] for result in client_results),
                "max_local_validation_accuracy": max(result["validation_metrics"]["accuracy"] for result in client_results),
                "min_local_validation_macro_f1": min(result["validation_metrics"]["macro_f1"] for result in client_results),
                "max_local_validation_macro_f1": max(result["validation_metrics"]["macro_f1"] for result in client_results),
                "fairness_accuracy_std": float(np.std([result["validation_metrics"]["accuracy"] for result in client_results])),
                "selection_threshold_accuracy": None,
                "selected_clients_current": list(range(1, 11)),
                "selected_clients_next": list(range(1, 11)),
                "selected_client_count": 10,
                "selected_client_count_next": 10,
                "global_validation_loss": float(
                    np.average(
                        [result["validation_metrics"]["loss"] for result in client_results],
                        weights=[result["validation_examples"] for result in client_results],
                    )
                ),
                "global_validation_accuracy": pooled_validation_metrics["accuracy"],
                "global_validation_macro_f1": pooled_validation_metrics["macro_f1"],
                "round_seconds": round_seconds,
                "cumulative_seconds": cumulative_seconds,
                "communication_bytes": logical_bytes,
                "communication_mib": logical_bytes / 1_048_576,
                "communication_cumulative_bytes": cumulative_communication,
                "communication_cumulative_mib": cumulative_communication / 1_048_576,
                "predicted_worker_load_gpu0_seconds": assignment["predicted_load_seconds"]["0"],
                "predicted_worker_load_gpu1_seconds": assignment["predicted_load_seconds"]["1"],
                "actual_worker_load_gpu0_seconds": worker_times["0"],
                "actual_worker_load_gpu1_seconds": worker_times["1"],
                "worker_idle_gpu0_seconds": worker_idle["0"],
                "worker_idle_gpu1_seconds": worker_idle["1"],
                "peak_allocated_gpu0_bytes": worker_results[0]["peak_allocated_bytes"],
                "peak_allocated_gpu1_bytes": worker_results[1]["peak_allocated_bytes"],
                "peak_reserved_gpu0_bytes": worker_results[0]["peak_reserved_bytes"],
                "peak_reserved_gpu1_bytes": worker_results[1]["peak_reserved_bytes"],
                "peak_allocated_max_bytes": max(result["peak_allocated_bytes"] for result in worker_results),
                "peak_allocated_sum_bytes": sum(result["peak_allocated_bytes"] for result in worker_results),
                "peak_reserved_max_bytes": max(result["peak_reserved_bytes"] for result in worker_results),
                "peak_reserved_sum_bytes": sum(result["peak_reserved_bytes"] for result in worker_results),
            }
            history_round.append(round_row)
            communication_rows.append(
                {
                    "round": round_index,
                    "selected_clients": 10,
                    "proxy_state_bytes": proxy_state_bytes,
                    "round_bytes": logical_bytes,
                    "round_mib": logical_bytes / 1_048_576,
                    "cumulative_bytes": cumulative_communication,
                    "cumulative_mib": cumulative_communication / 1_048_576,
                }
            )
            runtime_rounds.append(
                {"round": round_index, "seconds": round_seconds, "worker_seconds": worker_times}
            )
            checkpoint = {
                "model_state_dict": global_proxy_state,
                "global_proxy_state_dict": global_proxy_state,
                "personalized_model_state_dicts": personalized_states,
                "personalized_model_state_dicts_before_finetune": personalized_states,
                "historical_teacher_model_state_dicts": {},
                "initial_model_state_dicts_by_family": initial_model_state_dicts_by_family,
                "initialization_hashes": initialization_hashes,
                "initial_proxy_hash": initial_proxy_hash,
                "pretrained_global_state_hash": None,
                "global_state_hash": state_hash(global_proxy_state),
                "personalized_state_hashes": {
                    client_id: state_hash(state)
                    for client_id, state in personalized_states.items()
                },
                "round": round_index,
                "selected_clients_current_round": list(range(1, 11)),
                "selected_clients_next_round": list(range(1, 11)),
                "selection_threshold_accuracy": None,
                "local_accuracies": {
                    result["client_id"]: result["validation_metrics"]["accuracy"]
                    for result in client_results
                },
                "config": config,
                "feature_columns": FEATURE_COLUMNS,
                "label_mapping": label_mapping,
                "validation_metrics": {
                    "mean_personalized_accuracy": mean_accuracy,
                    "mean_personalized_macro_f1": mean_macro_f1,
                    "pooled_personalized": pooled_validation_metrics,
                },
                "model_metadata": model_metadata,
            }
            is_new_best = mean_macro_f1 > best_score
            if is_new_best:
                best_score = mean_macro_f1
                best_round = round_index
            write_table_pair(
                history_round,
                output_dir / "metrics" / "history_round.csv",
                output_dir / "metrics" / "history_round.json",
            )
            write_table_pair(
                history_client,
                output_dir / "metrics" / "history_client.csv",
                output_dir / "metrics" / "history_client.json",
            )
            write_table_pair(
                history_local_epoch,
                output_dir / "metrics" / "history_local_epoch.csv",
                output_dir / "metrics" / "history_local_epoch.json",
            )
            atomic_torch_save(checkpoint, output_dir / "checkpoints" / "last.pt")
            if is_new_best:
                atomic_torch_save(checkpoint, output_dir / "checkpoints" / "best.pt")
            logger.info(
                "Round %s complete: mean accuracy %.6f, mean macro-F1 %.6f",
                round_index,
                mean_accuracy,
                mean_macro_f1,
            )
        best_checkpoint = torch.load(
            output_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
        )
        final_test_started = time.perf_counter()
        for task_queue in task_queues:
            task_queue.put(
                {
                    "kind": "final_test",
                    "personalized_states": best_checkpoint["personalized_model_state_dicts"],
                    "test_paths": test_paths,
                }
            )
        test_worker_results = collect_worker_results(
            result_queue,
            workers,
            "test_complete",
            2,
            config["worker_timeout_seconds"],
        )
        test_results = []
        for result in test_worker_results:
            test_results.extend(
                torch.load(result["payload_path"], map_location="cpu", weights_only=False)
            )
        test_results.sort(key=lambda value: value["client_id"])
        if [result["client_id"] for result in test_results] != list(range(1, 11)):
            raise RuntimeError("Final test did not return exactly clients 1..10")
        pooled_confusion = np.sum(
            [result["confusion"] for result in test_results], axis=0, dtype=np.int64
        )
        expected_pooled_test_examples = int(test_paths["rows"]) * 10
        if int(pooled_confusion.sum()) != expected_pooled_test_examples:
            raise RuntimeError("Final pooled test sample accounting is incomplete")
        for result in test_results:
            if int(result["confusion"].sum()) != int(test_paths["rows"]):
                raise RuntimeError("A personalized test result has incomplete accounting")
        pooled_metrics = metrics_from_confusion(pooled_confusion)
        pooled_metrics["loss"] = float(
            np.mean([result["metrics"]["loss"] for result in test_results])
        )
        mean_personalized_metrics = {}
        scalar_metric_keys = [
            key
            for key in test_results[0]["metrics"]
            if not key.startswith("per_class_") and key != "examples"
        ]
        for key in scalar_metric_keys:
            mean_personalized_metrics[key] = float(
                np.mean([result["metrics"][key] for result in test_results])
            )
        report_json, report_rows = classification_report_from_metrics(
            pooled_metrics, label_mapping
        )
        write_json(output_dir / "metrics" / "classification_report.json", report_json)
        pd.DataFrame(report_rows).to_csv(
            output_dir / "metrics" / "classification_report.csv", index=False
        )
        pd.DataFrame(
            pooled_confusion,
            index=[label_mapping[index] for index in range(NUM_CLASSES)],
            columns=[label_mapping[index] for index in range(NUM_CLASSES)],
        ).to_csv(output_dir / "metrics" / "confusion_matrix.csv")
        np.save(output_dir / "metrics" / "confusion_matrix.npy", pooled_confusion)
        personalized_test_payload = []
        for result in test_results:
            client_report, _ = classification_report_from_metrics(
                result["metrics"], label_mapping
            )
            personalized_test_payload.append(
                {
                    "client_id": result["client_id"],
                    "model_family": result["model_family"],
                    "metrics": result["metrics"],
                    "classification_report": client_report,
                }
            )
        write_json(
            output_dir / "metrics" / "personalized_test_metrics.json",
            personalized_test_payload,
        )
        pd.DataFrame(
            [
                {
                    "client_id": result["client_id"],
                    "model_family": result["model_family"],
                    **{
                        key: value
                        for key, value in result["metrics"].items()
                        if not key.startswith("per_class_")
                    },
                }
                for result in test_results
            ]
        ).to_csv(output_dir / "metrics" / "personalized_test_metrics.csv", index=False)
        final_test_seconds = time.perf_counter() - final_test_started
        communication_payload = {
            "method": "pFedES",
            "logical_scope": "global proxy download plus local proxy upload only",
            "model_state_bytes": proxy_state_bytes,
            "total_bytes": cumulative_communication,
            "total_mib": cumulative_communication / 1_048_576,
            "ddp_gradient_allreduce_estimate": {
                "enabled": False,
                "bytes": 0,
                "mib": 0.0,
            },
            "rounds": communication_rows,
        }
        write_json(output_dir / "metrics" / "communication_costs.json", communication_payload)
        pd.DataFrame(communication_rows).to_csv(
            output_dir / "metrics" / "communication_costs.csv", index=False
        )
        sampler.stop()
        gpu_summary = {}
        for gpu_id in range(2):
            samples = [sample for sample in sampler.samples if sample["gpu_id"] == gpu_id]
            gpu_summary[str(gpu_id)] = {
                "sample_count": len(samples),
                "mean_utilization_percent": float(np.mean([sample["utilization_percent"] for sample in samples])) if samples else 0.0,
                "max_utilization_percent": float(np.max([sample["utilization_percent"] for sample in samples])) if samples else 0.0,
                "mean_memory_used_mib": float(np.mean([sample["memory_used_mib"] for sample in samples])) if samples else 0.0,
                "max_memory_used_mib": float(np.max([sample["memory_used_mib"] for sample in samples])) if samples else 0.0,
            }
        runtime_breakdown = {
            "data_validation_memmap_split_seconds": data_seconds,
            "server_pretrain_seconds": 0.0,
            "server_pretrain_enabled": False,
            "workload_and_stream_benchmark": stream_results,
            "gpu_cache_load_and_stream_benchmark_seconds_by_gpu": {
                str(result["gpu_id"]): result["initialization_seconds"]
                for result in initialization_results
            },
            "rounds": runtime_rounds,
            "federated_rounds_seconds": cumulative_seconds,
            "final_global_test_seconds": 0.0,
            "final_personalized_test_seconds": final_test_seconds,
            "gpu_utilization": gpu_summary,
            "total_entry_point_seconds_before_plotting": time.perf_counter() - pipeline_started,
        }
        write_json(output_dir / "metrics" / "runtime_breakdown.json", runtime_breakdown)
        summary = {
            "run_name": config["run_name"],
            "method": "pFedES",
            "model_family": config["scenario_model_family"],
            "proxy_model_family": "conv1d_1_8_1_same",
            "status": "complete",
            "config": config,
            "environment": environment,
            "model_metadata": model_metadata,
            "initialization_hashes": initialization_hashes,
            "initial_proxy_hash": initial_proxy_hash,
            "best_round": best_round,
            "best_mean_personalized_validation_macro_f1": best_score,
            "pretrain": {"enabled": False, "epochs": 0, "examples": 0},
            "dataset": dataset_summary,
            "client_assignment": assignment,
            "stream_benchmarks": stream_results,
            "cache": {
                "persistent_model_reserve_bytes_by_gpu": {
                    str(result["gpu_id"]): result["persistent_model_reserve_bytes"]
                    for result in initialization_results
                },
                "after_initialization": {
                    str(result["gpu_id"]): result["cache"]
                    for result in initialization_results
                },
                "after_final_test": {
                    str(result["gpu_id"]): result["cache"]
                    for result in test_worker_results
                },
            },
            "memory": {
                "peak_allocated_gpu0_bytes": max(row["peak_allocated_gpu0_bytes"] for row in history_round),
                "peak_allocated_gpu1_bytes": max(row["peak_allocated_gpu1_bytes"] for row in history_round),
                "peak_reserved_gpu0_bytes": max(row["peak_reserved_gpu0_bytes"] for row in history_round),
                "peak_reserved_gpu1_bytes": max(row["peak_reserved_gpu1_bytes"] for row in history_round),
            },
            "final_test_semantics": "each personalized model evaluates the full global test set; primary report pools all ten prediction sets",
            "final_pooled_personalized_test_metrics": pooled_metrics,
            "final_mean_personalized_test_metrics": mean_personalized_metrics,
            "final_personalized_test_metrics": personalized_test_payload,
            "communication": communication_payload,
            "runtime": runtime_breakdown,
            "gpu_utilization": gpu_summary,
            "outputs": {relative: relative for relative in REQUIRED_RELATIVE_OUTPUTS},
        }
        write_json(output_dir / "metrics" / "summary.json", summary)
        plotting_started = time.perf_counter()
        create_plots(
            output_dir,
            class_distribution,
            history_round,
            pooled_confusion,
            pooled_metrics,
            label_mapping,
        )
        runtime_breakdown["plotting_seconds"] = time.perf_counter() - plotting_started
        runtime_breakdown["total_entry_point_seconds"] = time.perf_counter() - pipeline_started
        write_json(output_dir / "metrics" / "runtime_breakdown.json", runtime_breakdown)
        summary["runtime"] = runtime_breakdown
        write_json(output_dir / "metrics" / "summary.json", summary)
    finally:
        sampler.stop()
        for task_queue in task_queues:
            task_queue.put({"kind": "shutdown"})
        for worker in workers:
            worker.join(timeout=30)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        bad_exit_codes = [worker.exitcode for worker in workers if worker.exitcode != 0]
        if bad_exit_codes:
            raise RuntimeError(f"Worker exit codes were nonzero: {bad_exit_codes}")
    verify_outputs(output_dir, config)
    logger.info("Run completed successfully: %s", config["run_name"])


if __name__ == "__main__":
    main()
