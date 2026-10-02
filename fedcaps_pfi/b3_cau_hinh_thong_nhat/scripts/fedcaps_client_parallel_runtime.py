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
import random
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
from torch import nn
from torch.amp import GradScaler, autocast


FEATURE_COLUMNS = [
    "ack_flag_number", "AVG", "Std", "UDP", "fin_count", "Max", "TCP",
    "syn_count", "Protocol Type", "Rate", "IAT", "syn_flag_number",
    "rst_flag_number", "Tot sum", "HTTPS", "ack_count", "fin_flag_number",
    "HTTP", "rst_count", "Header_Length", "psh_flag_number", "ICMP",
    "Time_To_Live", "ARP", "DNS",
]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["Label"]
NUM_CLASSES = 34
EXPECTED_PARAMETER_COUNTS = {"gru": 40034, "transformer": 71010, "cnn1d": 35874}
VALIDATED_MODEL_FAMILIES: set[str] = set()
COMMON_METRIC_NAMES = [
    "accuracy", "macro_precision", "micro_precision", "weighted_precision",
    "macro_recall", "micro_recall", "weighted_recall",
    "macro_f1", "micro_f1", "weighted_f1",
]


def read_manifest() -> dict:
    manifest_path = Path(os.environ["TRAINING_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest


def json_compatible(value):
    if isinstance(value, np.generic):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_compatible), encoding="utf-8")
    temporary.replace(path)


def rows_to_csv_json(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    atomic_json(json_path, rows)


def setup_logger(path: Path, name: str, console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def seed_from(base_seed: int, phase: str, epoch: int = 0, client_id: int = 0) -> int:
    digest = hashlib.sha256(f"{base_seed}|{phase}|{epoch}|{client_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class GRUClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(1, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x.unsqueeze(-1))
        return self.classifier(output[:, -1])


class TransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 64)
        self.position = nn.Parameter(torch.zeros(1, 25, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        encoded = self.projection(x.unsqueeze(-1)) + self.position[:, :length]
        return self.classifier(self.norm(self.encoder(encoded).mean(dim=1)))


class CNN1DClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.block3(self.block2(self.block1(x.unsqueeze(1))))
        return self.classifier(self.pool(hidden).squeeze(-1))


def build_classifier(family: str) -> nn.Module:
    if family == "gru":
        model = GRUClassifier()
    elif family == "transformer":
        model = TransformerClassifier()
    elif family == "cnn1d":
        model = CNN1DClassifier()
    else:
        raise ValueError(f"Unknown model family: {family}")
    if family not in VALIDATED_MODEL_FAMILIES:
        observed = count_parameters(model)
        expected = EXPECTED_PARAMETER_COUNTS[family]
        if observed != expected:
            raise AssertionError(f"{family} parameter count {observed} != {expected}")
        probe = torch.zeros(2, 25)
        if tuple(model(probe).shape) != (2, NUM_CLASSES):
            raise AssertionError(f"Invalid output shape for {family}")
        VALIDATED_MODEL_FAMILIES.add(family)
    return model


class MAB(nn.Module):
    def __init__(self, dimension: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.norm2 = nn.LayerNorm(dimension)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)
        hidden = self.norm1(query + attention)
        return self.norm2(hidden + self.feedforward(hidden))


class ISAB(nn.Module):
    def __init__(self, dimension: int = 128, heads: int = 4, inducing_points: int = 32) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, inducing_points, dimension) * 0.02)
        self.inducing_to_set = MAB(dimension, heads)
        self.set_to_inducing = MAB(dimension, heads)

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing.expand(values.shape[0], -1, -1)
        summary = self.inducing_to_set(inducing, values, values, key_padding_mask=padding_mask)
        output = self.set_to_inducing(values, summary, summary)
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(26, 128, padding_idx=0)
        self.isab1 = ISAB()
        self.isab2 = ISAB()

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        hidden = self.isab1(hidden, padding_mask)
        return self.isab2(hidden, padding_mask)


class SetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, 32, 128) * 0.02)
        self.pre = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.pma = MAB()
        self.self_mab = MAB()
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 25))

    def pool(self, embedding: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seeds = self.seeds.expand(embedding.shape[0], -1, -1)
        prepared = self.pre(embedding)
        pooled = self.pma(seeds, prepared, prepared, key_padding_mask=padding_mask)
        pooled = self.self_mab(pooled, pooled, pooled)
        return pooled.mean(dim=1)

    def logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)

    def forward(self, embedding: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.pool(embedding, padding_mask)
        return self.logits_from_latent(latent), latent


class FeatureQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class BatchedFeatureQNetwork(nn.Module):
    """Twenty-five independent Q networks evaluated with batched CUDA operations."""

    def __init__(self) -> None:
        super().__init__()
        independent = [FeatureQNetwork() for _ in range(25)]
        self.weight1 = nn.Parameter(torch.stack([network.network[0].weight.detach() for network in independent]))
        self.bias1 = nn.Parameter(torch.stack([network.network[0].bias.detach() for network in independent]))
        self.weight2 = nn.Parameter(torch.stack([network.network[2].weight.detach() for network in independent]))
        self.bias2 = nn.Parameter(torch.stack([network.network[2].bias.detach() for network in independent]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if tuple(state.shape) != (25,):
            raise ValueError(f"Expected one 25-feature MARLFS state, observed {tuple(state.shape)}")
        hidden = torch.einsum("fhi,i->fh", self.weight1, state) + self.bias1
        hidden = torch.relu(hidden)
        return torch.einsum("foh,fh->fo", self.weight2, hidden) + self.bias2


class GaussianActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Sequential(nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 128))
        self.log_std = nn.Parameter(torch.full((128,), -1.5))

    def distribution(self, latent: torch.Tensor):
        mean = self.mean(latent)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class RewardCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(25, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.network(mask).squeeze(-1)


def ensure_output_dirs(config: dict) -> dict[str, Path]:
    output = Path(config["output_dir"])
    paths = {
        "root": output,
        "checkpoints": output / "checkpoints",
        "logs": output / "logs",
        "metrics": output / "metrics",
        "artifacts": output / "artifacts",
        "temporary": Path(config["temporary_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fast_csv_rows(path: Path) -> int:
    line_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            line_count += block.count(b"\n")
    if path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                line_count += 1
    return max(0, line_count - 1)


def allocate_stratified(counts: np.ndarray, target: int, require_validation: bool) -> np.ndarray:
    counts = counts.astype(np.int64, copy=False)
    target = int(min(target, int(counts.sum())))
    if target <= 0:
        return np.zeros_like(counts)
    ideal = counts.astype(np.float64) * (target / max(1, int(counts.sum())))
    allocation = np.floor(ideal).astype(np.int64)
    present = np.flatnonzero(counts > 0)
    if target >= len(present):
        allocation[present] = np.maximum(allocation[present], 1)
    if require_validation:
        eligible = np.flatnonzero(counts >= 2)
        if target >= len(eligible):
            allocation[eligible] = np.maximum(allocation[eligible], 1)
    allocation = np.minimum(allocation, counts)
    while int(allocation.sum()) < target:
        candidates = np.flatnonzero(allocation < counts)
        residual = ideal[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmax(residual)])
        allocation[chosen] += 1
    while int(allocation.sum()) > target:
        minimum = np.where(counts > 0, 1, 0)
        candidates = np.flatnonzero(allocation > minimum)
        if len(candidates) == 0:
            break
        residual = ideal[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmin(residual)])
        allocation[chosen] -= 1
    return allocation


def convert_csv_to_memmap(csv_path: Path, prefix: Path, chunksize: int, logger: logging.Logger) -> dict:
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    if header != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected header for {csv_path}: {header}")
    rows = fast_csv_rows(csv_path)
    x_path = prefix.with_name(prefix.name + "_x.dat")
    y_path = prefix.with_name(prefix.name + "_y.dat")
    features = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(rows, 25))
    labels = np.memmap(y_path, mode="w+", dtype=np.int64, shape=(rows,))
    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if list(chunk.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Header changed while reading {csv_path}")
        x = chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        y_raw = chunk["Label"].to_numpy(copy=True)
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf in features: {csv_path}")
        if not np.isfinite(y_raw).all() or not np.equal(y_raw, np.floor(y_raw)).all():
            raise ValueError(f"Invalid labels: {csv_path}")
        y = y_raw.astype(np.int64, copy=False)
        if len(y) and (int(y.min()) < 0 or int(y.max()) >= NUM_CLASSES):
            raise ValueError(f"Label outside [0,33]: {csv_path}")
        end = offset + len(chunk)
        features[offset:end] = x
        labels[offset:end] = y
        offset = end
    if offset != rows:
        raise AssertionError(f"Row accounting mismatch for {csv_path}: {offset} != {rows}")
    features.flush()
    labels.flush()
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(np.int64)
    logger.info("Converted %s rows from %s", f"{rows:,}", csv_path.name)
    return {
        "rows": rows,
        "x_path": str(x_path),
        "y_path": str(y_path),
        "class_counts": counts.tolist(),
    }


def write_index_memmap(path: Path, values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.int64)
    target = np.memmap(path, mode="w+", dtype=np.int64, shape=(len(values),))
    target[:] = values
    target.flush()
    return str(path)


def make_client_splits(meta: dict, prefix: Path, seed: int, search_cap: int, folds: int) -> dict:
    """Use every local row for final training; CV exists only inside feature search."""
    rows = int(meta["rows"])
    labels = np.memmap(meta["y_path"], mode="r", dtype=np.int64, shape=(rows,))
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(np.int64)
    train_values = np.arange(rows, dtype=np.int64)
    train_path = prefix.with_name(prefix.name + "_train_idx.dat")
    write_index_memmap(train_path, train_values)

    search_total = min(int(search_cap), rows)
    search_counts = allocate_stratified(counts, search_total, False)
    pool_parts = []
    class_fold_parts = [[ ] for _ in range(folds)]
    for class_id in range(NUM_CLASSES):
        class_rows = np.flatnonzero(labels == class_id).astype(np.int64, copy=False)
        rng = np.random.default_rng(seed_from(seed, "fedcaps_search_pool", class_id, 0))
        rng.shuffle(class_rows)
        selected = class_rows[: int(search_counts[class_id])]
        pool_parts.append(selected)
        for fold_id, fold_rows in enumerate(np.array_split(selected, folds)):
            class_fold_parts[fold_id].append(np.asarray(fold_rows, dtype=np.int64))
    search_pool = np.concatenate(pool_parts) if pool_parts else np.empty(0, dtype=np.int64)
    pool_rng = np.random.default_rng(seed_from(seed, "fedcaps_search_pool_order"))
    pool_rng.shuffle(search_pool)
    search_pool_path = prefix.with_name(prefix.name + "_search_pool_idx.dat")
    write_index_memmap(search_pool_path, search_pool)

    fold_metadata = []
    for fold_id in range(folds):
        validation = np.concatenate(class_fold_parts[fold_id])
        validation_set = set(map(int, validation.tolist()))
        training = np.asarray([row for row in search_pool if int(row) not in validation_set], dtype=np.int64)
        fold_rng = np.random.default_rng(seed_from(seed, "fedcaps_search_fold", fold_id, 0))
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        train_name = f"search_fold_{fold_id}_train"
        validation_name = f"search_fold_{fold_id}_validation"
        train_fold_path = prefix.with_name(prefix.name + f"_{train_name}_idx.dat")
        validation_fold_path = prefix.with_name(prefix.name + f"_{validation_name}_idx.dat")
        write_index_memmap(train_fold_path, training)
        write_index_memmap(validation_fold_path, validation)
        meta[f"{train_name}_rows"] = len(training)
        meta[f"{train_name}_idx_path"] = str(train_fold_path)
        meta[f"{validation_name}_rows"] = len(validation)
        meta[f"{validation_name}_idx_path"] = str(validation_fold_path)
        fold_metadata.append({
            "fold": fold_id + 1,
            "train_rows": len(training),
            "validation_rows": len(validation),
        })
    meta.update({
        "train_rows": rows,
        "validation_rows": 0,
        "train_class_counts": counts.tolist(),
        "validation_class_counts": [0] * NUM_CLASSES,
        "train_idx_path": str(train_path),
        "search_pool_rows": len(search_pool),
        "search_pool_idx_path": str(search_pool_path),
        "search_cv_folds": folds,
        "search_fold_metadata": fold_metadata,
    })
    return meta


def validate_numpy_classification_cache(meta: dict) -> None:
    rows = int(meta["rows"])
    features = np.memmap(meta["x_path"], mode="r", dtype=np.float32, shape=(rows, 25))
    labels = np.memmap(meta["y_path"], mode="r", dtype=np.int64, shape=(rows,))
    if features.shape != (rows, 25) or features.dtype != np.float32:
        raise AssertionError("Invalid cached feature contract")
    if labels.shape != (rows,) or labels.dtype != np.int64:
        raise AssertionError("Invalid cached label contract")
    if rows and (int(labels.min()) < 0 or int(labels.max()) >= NUM_CLASSES):
        raise AssertionError("Cached label outside [0,33]")
    index_contracts = [("train_idx_path", "train_rows"), ("search_pool_idx_path", "search_pool_rows")]
    for fold_id in range(int(meta["search_cv_folds"])):
        index_contracts.extend([
            (f"search_fold_{fold_id}_train_idx_path", f"search_fold_{fold_id}_train_rows"),
            (f"search_fold_{fold_id}_validation_idx_path", f"search_fold_{fold_id}_validation_rows"),
        ])
    for name, count_name in index_contracts:
        count = int(meta[count_name])
        indices = np.memmap(meta[name], mode="r", dtype=np.int64, shape=(count,))
        if count and (int(indices.min()) < 0 or int(indices.max()) >= rows):
            raise AssertionError(f"Invalid index range: {name}")
    if int(meta["train_rows"]) != rows or int(meta["validation_rows"]) != 0:
        raise AssertionError("FedCAPS final training must use 100% local data and no local validation")
    for fold in meta["search_fold_metadata"]:
        if int(fold["train_rows"]) + int(fold["validation_rows"]) != int(meta["search_pool_rows"]):
            raise AssertionError("FedCAPS CV fold accounting mismatch")


def prepare_data(config: dict, paths: dict[str, Path], logger: logging.Logger) -> dict:
    started = time.perf_counter()
    input_dir = Path(config["input_dir"])
    logger.info("[STAGE data] Validating input contract under %s", input_dir)
    required = [input_dir / f"client_{client_id}_train.csv" for client_id in range(1, 11)]
    required += [input_dir / "global_train_data.csv", input_dir / "global_test_data.csv", input_dir / "label_mapping.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    mapping = pd.read_csv(input_dir / "label_mapping.csv")
    if list(mapping.columns) != ["Encoded_ID", "Label_Name"] or len(mapping) != NUM_CLASSES:
        raise ValueError("Invalid label_mapping.csv")
    encoded = mapping["Encoded_ID"].to_numpy(dtype=np.int64)
    if not np.array_equal(encoded, np.arange(NUM_CLASSES, dtype=np.int64)):
        raise ValueError("Label mapping IDs must be 0..33")
    cache_root = paths["temporary"] / "data"
    cache_root.mkdir(parents=True, exist_ok=True)
    clients = {}
    distribution_rows = []
    for client_id in range(1, 11):
        client_started = time.perf_counter()
        logger.info("[STAGE data] Preparing client %d/10", client_id)
        prefix = cache_root / f"client_{client_id}"
        meta = convert_csv_to_memmap(
            input_dir / f"client_{client_id}_train.csv",
            prefix,
            int(config["csv_chunksize"]),
            logger,
        )
        meta = make_client_splits(
            meta,
            prefix,
            int(config["seed"]),
            int(config["fedcaps_search_pool_max_rows_per_client"]),
            int(config["fedcaps_search_cv_folds"]),
        )
        meta["client_id"] = client_id
        meta["model_family"] = config["client_models"][str(client_id)]
        validate_numpy_classification_cache(meta)
        clients[str(client_id)] = meta
        logger.info(
            "[STAGE data] Client %d ready rows=%s final_train=%s local_validation=%s search_pool=%s cv_folds=%s seconds=%.2f",
            client_id,
            f"{int(meta['rows']):,}",
            f"{int(meta['train_rows']):,}",
            f"{int(meta['validation_rows']):,}",
            f"{int(meta['search_pool_rows']):,}",
            int(meta["search_cv_folds"]),
            time.perf_counter() - client_started,
        )
        for class_id, count in enumerate(meta["class_counts"]):
            distribution_rows.append({
                "client_id": client_id,
                "class_id": class_id,
                "class_name": str(mapping.loc[class_id, "Label_Name"]),
                "count": int(count),
                "train_count": int(meta["train_class_counts"][class_id]),
                "validation_count": int(meta["validation_class_counts"][class_id]),
            })
    global_test = convert_csv_to_memmap(
        input_dir / "global_test_data.csv",
        cache_root / "global_test",
        int(config["csv_chunksize"]),
        logger,
    )
    global_train_rows = fast_csv_rows(input_dir / "global_train_data.csv")
    local_rows = sum(int(meta["rows"]) for meta in clients.values())
    if global_train_rows != local_rows:
        raise AssertionError(f"global_train rows {global_train_rows} != client sum {local_rows}")
    dataset_summary = {
        "feature_columns": FEATURE_COLUMNS,
        "num_features": 25,
        "num_classes": NUM_CLASSES,
        "label_mapping": mapping.to_dict(orient="records"),
        "global_train_rows": global_train_rows,
        "client_rows_sum": local_rows,
        "global_test": global_test,
        "clients": clients,
        "split_seed": int(config["seed"]),
        "split_policy": "100_percent_local_training; deterministic_stratified_5_fold_cv_only_for_fedcaps_feature_search",
        "fedcaps_search_cv_folds": int(config["fedcaps_search_cv_folds"]),
        "fedcaps_search_pool_max_rows_per_client": int(config["fedcaps_search_pool_max_rows_per_client"]),
        "preprocessing_seconds": time.perf_counter() - started,
    }
    atomic_json(paths["metrics"] / "dataset_summary.json", dataset_summary)
    pd.DataFrame(distribution_rows).to_csv(paths["metrics"] / "client_class_distribution.csv", index=False)
    logger.info(
        "[STAGE data] Complete local_rows=%s global_test_rows=%s seconds=%.2f",
        f"{local_rows:,}",
        f"{int(global_test['rows']):,}",
        time.perf_counter() - started,
    )
    return dataset_summary


def metrics_from_confusion(confusion: np.ndarray, loss_sum: float = 0.0, examples: int | None = None) -> dict:
    matrix = confusion.astype(np.float64, copy=False)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    total = float(matrix.sum())
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    precision = np.divide(true_positive, true_positive + false_positive, out=np.zeros(NUM_CLASSES), where=(true_positive + false_positive) > 0)
    recall = np.divide(true_positive, true_positive + false_negative, out=np.zeros(NUM_CLASSES), where=(true_positive + false_negative) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros(NUM_CLASSES), where=(precision + recall) > 0)
    weights = np.divide(support, total, out=np.zeros(NUM_CLASSES), where=total > 0)
    accuracy = float(true_positive.sum() / total) if total else 0.0
    denominator = examples if examples is not None else int(total)
    return {
        "loss": float(loss_sum / denominator) if denominator else 0.0,
        "accuracy": accuracy,
        "micro_precision": accuracy,
        "micro_recall": accuracy,
        "micro_f1": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "examples": int(total),
    }


def classification_report_from_confusion(confusion: np.ndarray, label_mapping: list[dict]) -> tuple[list[dict], dict]:
    matrix = confusion.astype(np.float64, copy=False)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros(NUM_CLASSES), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(NUM_CLASSES), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(NUM_CLASSES), where=(precision + recall) > 0)
    total = float(support.sum())
    weights = np.divide(support, total, out=np.zeros(NUM_CLASSES), where=total > 0)
    rows = []
    for class_id in range(NUM_CLASSES):
        rows.append({
            "class_id": class_id,
            "class_name": str(label_mapping[class_id]["Label_Name"]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        })
    aggregate = {
        "macro_avg": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean()), "support": int(total)},
        "weighted_avg": {"precision": float(np.sum(precision * weights)), "recall": float(np.sum(recall * weights)), "f1": float(np.sum(f1 * weights)), "support": int(total)},
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
    }
    return rows, aggregate


def tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element = torch.empty((), dtype=dtype).element_size()
    return int(math.prod(shape) * element)


class GPUClientCache:
    def __init__(self, gpu_id: int, config: dict, clients: dict, logger: logging.Logger, phase: str) -> None:
        self.gpu_id = gpu_id
        self.device = torch.device("cuda", gpu_id)
        self.config = config
        self.clients = clients
        self.logger = logger
        self.phase = phase
        self.owned_client_ids: list[int] = []
        properties = torch.cuda.get_device_properties(gpu_id)
        self.cache_budget_bytes = int(properties.total_memory * float(config["gpu_cache_fraction"]))
        self.entries: OrderedDict[int, dict] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.fallbacks = 0
        self.actual_cache_bytes = 0
        self.auxiliary_cache_bytes = 0
        self.test_cache_hits = 0
        self.test_cache_misses = 0
        self.test_entry: dict | None = None
        self.fallback_entries: dict[int, dict] = {}
        self.model_pools: dict[str, list[nn.Module]] = {}

    def planned_bytes(self, meta: dict) -> int:
        rows = int(meta["rows"])
        index_rows = int(meta["train_rows"])
        for fold_id in range(int(meta["search_cv_folds"])):
            index_rows += int(meta[f"search_fold_{fold_id}_train_rows"])
            index_rows += int(meta[f"search_fold_{fold_id}_validation_rows"])
        return tensor_bytes((rows, 25), torch.float32) + tensor_bytes((rows,), torch.long) + tensor_bytes((index_rows,), torch.long)

    def _copy_memmap(self, path: str, dtype, shape: tuple[int, ...], torch_dtype: torch.dtype) -> torch.Tensor:
        source = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        target = torch.empty(shape, dtype=torch_dtype, device=self.device)
        chunk_rows = int(self.config["gpu_copy_chunk_rows"])
        for start in range(0, shape[0], chunk_rows):
            end = min(shape[0], start + chunk_rows)
            host = torch.from_numpy(np.array(source[start:end], copy=True))
            target[start:end].copy_(host, non_blocking=False)
        return target

    def _load_full(self, client_id: int) -> dict:
        meta = self.clients[str(client_id)]
        rows = int(meta["rows"])
        entry = {
            "mode": "gpu_full",
            "meta": meta,
            "x": self._copy_memmap(meta["x_path"], np.float32, (rows, 25), torch.float32),
            "y": self._copy_memmap(meta["y_path"], np.int64, (rows,), torch.long),
            "indices": {},
        }
        split_names = ["train"]
        for fold_id in range(int(meta["search_cv_folds"])):
            split_names.extend([f"search_fold_{fold_id}_train", f"search_fold_{fold_id}_validation"])
        for name in split_names:
            count_name = f"{name}_rows"
            count = int(meta[count_name])
            entry["indices"][name] = self._copy_memmap(meta[f"{name}_idx_path"], np.int64, (count,), torch.long)
        entry["bytes"] = self.planned_bytes(meta)
        return entry

    def _pinned_fallback_entry(self, meta: dict) -> dict:
        slots = max(1, int(self.config["runtime_stream_count_by_gpu"][str(self.gpu_id)]))
        batch_size = int(self.config["per_client_batch_size"])
        return {
            "mode": "pinned_fallback",
            "meta": meta,
            "pinned_x": [
                torch.empty((batch_size, 25), dtype=torch.float32, pin_memory=True)
                for _ in range(slots)
            ],
            "pinned_y": [
                torch.empty((batch_size,), dtype=torch.long, pin_memory=True)
                for _ in range(slots)
            ],
        }

    def get_model_pool(self, family: str, count: int) -> list[nn.Module]:
        models = self.model_pools.setdefault(family, [])
        while len(models) < count:
            models.append(build_classifier(family).to(self.device))
        return models[:count]

    def initialize(self, owned_client_ids: list[int]) -> dict:
        self.owned_client_ids = sorted(map(int, owned_client_ids))
        planned = sum(self.planned_bytes(self.clients[str(client_id)]) for client_id in owned_client_ids)
        if planned <= self.cache_budget_bytes:
            for client_id in owned_client_ids:
                entry = self._load_full(client_id)
                self.entries[client_id] = entry
                self.actual_cache_bytes += int(entry["bytes"])
        else:
            ordered = sorted(owned_client_ids, key=lambda client_id: (self.planned_bytes(self.clients[str(client_id)]), client_id))
            for client_id in ordered:
                needed = self.planned_bytes(self.clients[str(client_id)])
                if self.actual_cache_bytes + needed <= self.cache_budget_bytes:
                    entry = self._load_full(client_id)
                    self.entries[client_id] = entry
                    self.actual_cache_bytes += int(entry["bytes"])
                else:
                    self.fallbacks += 1
        torch.cuda.synchronize(self.device)
        self.logger.info(
            "GPU %d cache phase=%s clients=%s planned=%s actual=%s budget=%s",
            self.gpu_id,
            self.phase,
            self.owned_client_ids,
            f"{planned:,}",
            f"{self.actual_cache_bytes:,}",
            f"{self.cache_budget_bytes:,}",
        )
        return self.telemetry(planned)

    def get(self, client_id: int) -> dict:
        if client_id in self.entries:
            self.cache_hits += 1
            self.entries.move_to_end(client_id)
            return self.entries[client_id]
        self.cache_misses += 1
        needed = self.planned_bytes(self.clients[str(client_id)])
        while self.entries and self.actual_cache_bytes + self.auxiliary_cache_bytes + needed > self.cache_budget_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.actual_cache_bytes -= int(evicted["bytes"])
            self.evictions += 1
            del evicted
            torch.cuda.empty_cache()
        if self.auxiliary_cache_bytes + needed <= self.cache_budget_bytes:
            entry = self._load_full(client_id)
            self.entries[client_id] = entry
            self.actual_cache_bytes += int(entry["bytes"])
            return entry
        self.fallbacks += 1
        if client_id not in self.fallback_entries:
            self.fallback_entries[client_id] = self._pinned_fallback_entry(self.clients[str(client_id)])
        return self.fallback_entries[client_id]

    def get_global_test(self, global_test_meta: dict) -> dict:
        if self.test_entry is not None:
            self.test_cache_hits += 1
            return self.test_entry
        self.test_cache_misses += 1
        rows = int(global_test_meta["rows"])
        test_bytes = tensor_bytes((rows, 25), torch.float32) + 2 * tensor_bytes((rows,), torch.long)
        if (
            bool(self.config["persistent_global_test_gpu_cache"])
            and self.actual_cache_bytes + test_bytes <= self.cache_budget_bytes
        ):
            self.test_entry = {
                "mode": "gpu_full",
                "meta": {
                    **global_test_meta,
                    "test_rows": rows,
                },
                "x": self._copy_memmap(global_test_meta["x_path"], np.float32, (rows, 25), torch.float32),
                "y": self._copy_memmap(global_test_meta["y_path"], np.int64, (rows,), torch.long),
                "indices": {"test": torch.arange(rows, dtype=torch.long, device=self.device)},
            }
            self.auxiliary_cache_bytes = test_bytes
        else:
            identity_path = Path(global_test_meta["x_path"]).with_name(
                f"global_test_identity_idx_gpu_{self.gpu_id}.dat"
            )
            if not identity_path.is_file() or identity_path.stat().st_size != rows * np.dtype(np.int64).itemsize:
                identity = np.memmap(identity_path, mode="w+", dtype=np.int64, shape=(rows,))
                identity[:] = np.arange(rows, dtype=np.int64)
                identity.flush()
            self.test_entry = self._pinned_fallback_entry({
                **global_test_meta,
                "test_rows": rows,
                "test_idx_path": str(identity_path),
            })
        torch.cuda.synchronize(self.device)
        self.logger.info(
            "GPU %d global-test cache mode=%s rows=%s auxiliary_bytes=%s",
            self.gpu_id,
            self.test_entry["mode"],
            f"{rows:,}",
            f"{self.auxiliary_cache_bytes:,}",
        )
        return self.test_entry

    def telemetry(self, planned: int | None = None) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "phase": self.phase,
            "owned_client_ids": self.owned_client_ids,
            "cache_budget_bytes": self.cache_budget_bytes,
            "planned_cache_bytes": int(planned if planned is not None else self.actual_cache_bytes),
            "actual_cache_bytes": self.actual_cache_bytes,
            "auxiliary_cache_bytes": self.auxiliary_cache_bytes,
            "test_cache_hits": self.test_cache_hits,
            "test_cache_misses": self.test_cache_misses,
            "resident_model_pool_sizes": {
                family: len(models) for family, models in sorted(self.model_pools.items())
            },
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "fallbacks": self.fallbacks,
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }


def make_training_context(device: torch.device, train_examples: int, seed: int) -> dict:
    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        order = torch.randperm(train_examples, generator=generator, device=device)
        loss_sums = torch.zeros(4, dtype=torch.float64, device=device)
    return {"stream": stream, "generator": generator, "order": order, "loss_sums": loss_sums}


def bn_singleton_mode(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval() if enabled else module.train()


def batch_from_entry(
    entry: dict,
    split: str,
    positions: torch.Tensor,
    feature_ids: torch.Tensor,
    device: torch.device,
    staging_slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if entry["mode"] == "gpu_full":
        row_ids = entry["indices"][split].index_select(0, positions)
        x = entry["x"].index_select(0, row_ids).index_select(1, feature_ids)
        y = entry["y"].index_select(0, row_ids)
        return x, y
    meta = entry["meta"]
    count = int(meta[f"{split}_rows"])
    index_map = np.memmap(meta[f"{split}_idx_path"], mode="r", dtype=np.int64, shape=(count,))
    if positions.device.type != "cpu" or feature_ids.device != device:
        raise AssertionError("Pinned fallback requires CPU positions and device-owned feature IDs")
    position_numpy = positions.numpy()
    row_ids = np.array(index_map[position_numpy], copy=True)
    rows = int(meta["rows"])
    features = np.memmap(meta["x_path"], mode="r", dtype=np.float32, shape=(rows, 25))
    labels = np.memmap(meta["y_path"], mode="r", dtype=np.int64, shape=(rows,))
    batch_rows = len(row_ids)
    pinned_x = entry["pinned_x"][staging_slot][:batch_rows]
    pinned_y = entry["pinned_y"][staging_slot][:batch_rows]
    pinned_x.copy_(torch.from_numpy(np.array(features[row_ids], copy=True)))
    pinned_y.copy_(torch.from_numpy(np.array(labels[row_ids], copy=True)))
    # Reused bounded pinned buffers must complete their H2D copy before the CPU
    # overwrites them for the next batch.
    x = pinned_x.to(device, non_blocking=False).index_select(1, feature_ids)
    y = pinned_y.to(device, non_blocking=False)
    return x, y


def evaluate_model(
    model: nn.Module,
    entry: dict,
    split: str,
    feature_ids_list: list[int],
    batch_size: int,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[np.ndarray, float, int]:
    model.eval()
    count = int(entry["meta"][f"{split}_rows"])
    with torch.cuda.stream(stream):
        index_device = device if entry["mode"] == "gpu_full" else torch.device("cpu")
        feature_ids = torch.tensor(feature_ids_list, dtype=torch.long, device=device)
        confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for start in range(0, count, batch_size):
                end = min(count, start + batch_size)
                positions = torch.arange(start, end, dtype=torch.long, device=index_device)
                x, y = batch_from_entry(entry, split, positions, feature_ids, device)
                with autocast("cuda"):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, reduction="sum")
                prediction = logits.argmax(dim=1)
                encoded = y * NUM_CLASSES + prediction
                confusion += torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
                loss_sum += loss.detach().double()
    stream.synchronize()
    return confusion.cpu().numpy(), float(loss_sum.cpu()), count


def evaluate_candidate(
    cache: GPUClientCache,
    client_id: int,
    family: str,
    feature_ids_list: list[int],
    initial_state: dict,
    config: dict,
    phase_seed: int,
) -> dict:
    device = cache.device
    entry = cache.get(client_id)
    started = time.perf_counter()
    folds = int(config["fedcaps_search_cv_folds"])
    stream_count = min(folds, int(config["runtime_stream_count_by_gpu"][str(cache.gpu_id)]))
    pooled_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    validation_loss_sum = 0.0
    validation_examples = 0
    training_loss_sum = 0.0
    training_examples = 0
    fold_rows = []
    torch.cuda.manual_seed(seed_from(phase_seed, "candidate_cuda", 0, client_id))
    batch_size = int(config["per_client_batch_size"])
    model_slots = cache.get_model_pool(family, stream_count)
    for wave_start in range(0, folds, stream_count):
        wave = []
        for fold_id in range(wave_start, min(folds, wave_start + stream_count)):
            fold_seed = seed_from(phase_seed, "fedcaps_cv", fold_id, client_id)
            stream_slot = fold_id - wave_start
            model = model_slots[stream_slot]
            model.load_state_dict(initial_state)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=float(config["classifier_learning_rate"]),
                foreach=True,
            )
            scaler = GradScaler("cuda")
            train_split = f"search_fold_{fold_id}_train"
            validation_split = f"search_fold_{fold_id}_validation"
            train_examples = int(entry["meta"][f"{train_split}_rows"])
            context = make_training_context(device, train_examples, fold_seed)
            if entry["mode"] == "gpu_full":
                training_order = context["order"]
                index_device = device
            else:
                cpu_generator = torch.Generator().manual_seed(fold_seed)
                training_order = torch.randperm(train_examples, generator=cpu_generator)
                index_device = torch.device("cpu")
            with torch.cuda.stream(context["stream"]):
                feature_ids = torch.tensor(feature_ids_list, dtype=torch.long, device=device)
            model.train()
            wave.append({
                "fold_id": fold_id,
                "model": model,
                "optimizer": optimizer,
                "scaler": scaler,
                "train_split": train_split,
                "validation_split": validation_split,
                "train_examples": train_examples,
                "training_order": training_order,
                "index_device": index_device,
                "feature_ids": feature_ids,
                "staging_slot": stream_slot,
                **context,
            })

        maximum_train_batches = max(math.ceil(context["train_examples"] / batch_size) for context in wave)
        for batch_id in range(maximum_train_batches):
            start = batch_id * batch_size
            for context in wave:
                if start >= context["train_examples"]:
                    continue
                end = min(context["train_examples"], start + batch_size)
                with torch.cuda.stream(context["stream"]):
                    positions = context["training_order"][start:end]
                    x, y = batch_from_entry(
                        entry,
                        context["train_split"],
                        positions,
                        context["feature_ids"],
                        device,
                        context["staging_slot"],
                    )
                    singleton = family == "cnn1d" and len(feature_ids_list) < 4 and x.shape[0] == 1
                    if singleton:
                        bn_singleton_mode(context["model"], True)
                    context["optimizer"].zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        logits = context["model"](x)
                        loss = nn.functional.cross_entropy(logits, y)
                    context["scaler"].scale(loss).backward()
                    context["scaler"].step(context["optimizer"])
                    context["scaler"].update()
                    context["loss_sums"][0] += loss.detach().double() * x.shape[0]
                    context["loss_sums"][1] += x.shape[0]
                    if singleton:
                        bn_singleton_mode(context["model"], False)
        for context in wave:
            context["stream"].synchronize()

        for context in wave:
            context["model"].eval()
            fold_examples = int(entry["meta"][f"{context['validation_split']}_rows"])
            with torch.cuda.stream(context["stream"]):
                context["confusion"] = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
                context["validation_loss_sum"] = torch.zeros((), dtype=torch.float64, device=device)
            context["fold_examples"] = fold_examples
        maximum_validation_batches = max(math.ceil(context["fold_examples"] / batch_size) for context in wave)
        for batch_id in range(maximum_validation_batches):
            start = batch_id * batch_size
            for context in wave:
                if start >= context["fold_examples"]:
                    continue
                end = min(context["fold_examples"], start + batch_size)
                with torch.cuda.stream(context["stream"]):
                    positions = torch.arange(start, end, dtype=torch.long, device=context["index_device"])
                    x, y = batch_from_entry(
                        entry,
                        context["validation_split"],
                        positions,
                        context["feature_ids"],
                        device,
                        context["staging_slot"],
                    )
                    with torch.no_grad(), autocast("cuda"):
                        logits = context["model"](x)
                        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
                    prediction = logits.argmax(dim=1)
                    encoded = y * NUM_CLASSES + prediction
                    context["confusion"] += torch.bincount(
                        encoded,
                        minlength=NUM_CLASSES * NUM_CLASSES,
                    ).reshape(NUM_CLASSES, NUM_CLASSES)
                    context["validation_loss_sum"] += loss.detach().double()
        for context in wave:
            context["stream"].synchronize()

        for context in wave:
            fold_confusion = context["confusion"].cpu().numpy()
            fold_loss_sum = float(context["validation_loss_sum"].cpu())
            fold_examples = int(context["fold_examples"])
            pooled_confusion += fold_confusion
            validation_loss_sum += fold_loss_sum
            validation_examples += fold_examples
            fold_training_examples = int(float(context["loss_sums"][1].cpu()))
            fold_training_loss = float(context["loss_sums"][0].cpu())
            training_loss_sum += fold_training_loss
            training_examples += fold_training_examples
            fold_metrics = metrics_from_confusion(fold_confusion, fold_loss_sum, fold_examples)
            fold_rows.append({
                "fold": int(context["fold_id"]) + 1,
                "train_examples": int(context["train_examples"]),
                "validation_examples": fold_examples,
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "micro_f1": fold_metrics["micro_f1"],
            })
    fold_rows.sort(key=lambda row: int(row["fold"]))
    metrics = metrics_from_confusion(pooled_confusion, validation_loss_sum, validation_examples)
    return {
        "client_id": client_id,
        "model_family": family,
        "feature_ids": feature_ids_list,
        "subset_size": len(feature_ids_list),
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "validation_loss": metrics["loss"],
        "train_loss": training_loss_sum / max(1, training_examples),
        "cv_folds": folds,
        "stream_count": stream_count,
        "fold_metrics": fold_rows,
        "seconds": time.perf_counter() - started,
        "gpu_id": cache.gpu_id,
    }


def execute_benchmark_family(
    family: str,
    config: dict,
    device: torch.device,
    stream_count: int,
    seed: int,
) -> tuple[float, list[dict[str, torch.Tensor]]]:
    set_all_seeds(seed)
    torch.cuda.manual_seed_all(seed)
    models = []
    optimizers = []
    scalers = []
    streams = []
    batches = []
    labels = []
    for stream_id in range(stream_count):
        model = build_classifier(family).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config["classifier_learning_rate"]),
            foreach=True,
        )
        scaler = GradScaler("cuda")
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            x = torch.randn(int(config["per_client_batch_size"]), 25, device=device)
            y = torch.randint(0, NUM_CLASSES, (int(config["per_client_batch_size"]),), device=device)
        models.append(model)
        optimizers.append(optimizer)
        scalers.append(scaler)
        streams.append(stream)
        batches.append(x)
        labels.append(y)
    default_stream = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(default_stream)
    for warmup in range(2):
        for stream_id, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                optimizers[stream_id].zero_grad(set_to_none=True)
                with autocast("cuda"):
                    loss = nn.functional.cross_entropy(models[stream_id](batches[stream_id]), labels[stream_id])
                scalers[stream_id].scale(loss).backward()
                scalers[stream_id].step(optimizers[stream_id])
                scalers[stream_id].update()
    for stream in streams:
        stream.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    finish_event = torch.cuda.Event(enable_timing=True)
    end_events = [torch.cuda.Event(enable_timing=True) for _ in streams]
    start_event.record(default_stream)
    iterations = 8
    for measured in range(iterations):
        for stream_id, stream in enumerate(streams):
            stream.wait_event(start_event)
            with torch.cuda.stream(stream):
                optimizers[stream_id].zero_grad(set_to_none=True)
                with autocast("cuda"):
                    loss = nn.functional.cross_entropy(models[stream_id](batches[stream_id]), labels[stream_id])
                scalers[stream_id].scale(loss).backward()
                scalers[stream_id].step(optimizers[stream_id])
                scalers[stream_id].update()
    for stream_id, stream in enumerate(streams):
        end_events[stream_id].record(stream)
        default_stream.wait_event(end_events[stream_id])
    finish_event.record(default_stream)
    torch.cuda.synchronize(device)
    seconds = float(start_event.elapsed_time(finish_event)) / 1000.0
    states = [cpu_state(model) for model in models]
    return seconds, states


def benchmark_family(family: str, config: dict, device: torch.device, stream_count: int) -> dict:
    seed = seed_from(int(config["seed"]), "benchmark", stream_count, EXPECTED_PARAMETER_COUNTS[family])
    first_seconds, first_states = execute_benchmark_family(family, config, device, stream_count, seed)
    reproducible = True
    replay_max_abs_difference = 0.0
    first_hashes = [state_hash(state) for state in first_states]
    replay_hashes = first_hashes
    seconds = first_seconds
    if stream_count > 1:
        replay_seconds, replay_states = execute_benchmark_family(family, config, device, stream_count, seed)
        replay_hashes = [state_hash(state) for state in replay_states]
        reproducible = first_hashes == replay_hashes
        for first_state, replay_state in zip(first_states, replay_states):
            for name in first_state:
                difference = (first_state[name].to(torch.float64) - replay_state[name].to(torch.float64)).abs().max()
                replay_max_abs_difference = max(replay_max_abs_difference, float(difference))
        seconds = min(first_seconds, replay_seconds)
    measured_iterations = 8
    samples = measured_iterations * stream_count * int(config["per_client_batch_size"])
    return {
        "family": family,
        "stream_count": stream_count,
        "seconds": seconds,
        "samples": samples,
        "samples_per_second": samples / max(seconds, 1e-9),
        "seconds_per_step": seconds / max(1, measured_iterations * stream_count),
        "deterministic_replay_valid": reproducible,
        "deterministic_replay_max_abs_difference": replay_max_abs_difference,
        "state_hashes": first_hashes,
        "replay_state_hashes": replay_hashes,
    }


def collect_marlfs_records(
    cache: GPUClientCache,
    client_ids: list[int],
    initial_states: dict,
    config: dict,
    output_path: Path,
) -> dict:
    records = []
    started = time.perf_counter()
    for client_id in client_ids:
        family = config["client_models"][str(client_id)]
        client_started = time.perf_counter()
        cache.logger.info(
            "[STAGE MARLFS] GPU %d client=%d family=%s start epochs=%d",
            cache.gpu_id,
            client_id,
            family,
            int(config["marlfs_epochs"]),
        )
        rng = np.random.default_rng(seed_from(int(config["seed"]), "marlfs", 0, client_id))
        agents = BatchedFeatureQNetwork().to(cache.device)
        optimizer = torch.optim.Adam(agents.parameters(), lr=0.001)
        state = np.ones(25, dtype=np.float32)
        baseline = 0.0
        for epoch in range(1, int(config["marlfs_epochs"]) + 1):
            epsilon = max(0.05, 1.0 - epoch / max(1.0, 0.80 * float(config["marlfs_epochs"])))
            state_tensor = torch.from_numpy(state).to(cache.device)
            with torch.no_grad():
                greedy_actions = agents(state_tensor).argmax(dim=1).cpu().numpy()
            actions = []
            for feature_id in range(25):
                if float(rng.random()) < epsilon:
                    action = int(rng.integers(0, 2))
                else:
                    action = int(greedy_actions[feature_id])
                actions.append(action)
            selected = np.flatnonzero(np.asarray(actions, dtype=np.int64)).tolist()
            if len(selected) < int(config["minimum_subset_size"]):
                ranking = rng.permutation(25).tolist()
                selected = sorted(ranking[: int(config["minimum_subset_size"])])
            evaluation = evaluate_candidate(
                cache,
                client_id,
                family,
                selected,
                initial_states[family],
                config,
                seed_from(int(config["seed"]), "marlfs_candidate", epoch, client_id),
            )
            performance = float(evaluation["micro_f1"])
            compactness = 1.0 - len(selected) / 25.0
            reward = performance + 0.05 * compactness
            baseline = 0.95 * baseline + 0.05 * reward
            next_state = np.zeros(25, dtype=np.float32)
            next_state[selected] = 1.0
            next_tensor = torch.from_numpy(next_state).to(cache.device)
            action_tensor = torch.tensor(actions, dtype=torch.long, device=cache.device)
            q_values = agents(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_values = agents(next_tensor).max(dim=1).values
                targets = torch.full_like(next_values, reward - baseline) + 0.99 * next_values
            optimizer.zero_grad(set_to_none=True)
            q_loss = nn.functional.smooth_l1_loss(q_values, targets)
            q_loss.backward()
            optimizer.step()
            records.append({
                "record_id": f"c{client_id:02d}_r{epoch:03d}",
                "client_id": client_id,
                "model_family": family,
                "collection_epoch": epoch,
                "feature_ids": selected,
                "feature_names": [FEATURE_COLUMNS[feature_id] for feature_id in selected],
                "subset_size": len(selected),
                "performance_micro_f1": performance,
                "performance_macro_f1": float(evaluation["macro_f1"]),
                "collector_reward": reward,
                "q_loss": float(q_loss.detach().cpu()),
                "train_examples": int(cache.clients[str(client_id)]["train_rows"]),
                "search_pool_examples": int(cache.clients[str(client_id)]["search_pool_rows"]),
                "search_cv_folds": int(cache.clients[str(client_id)]["search_cv_folds"]),
                "gpu_id": cache.gpu_id,
                "seconds": float(evaluation["seconds"]),
            })
            if epoch == 1 or epoch % int(config["marlfs_log_interval"]) == 0 or epoch == int(config["marlfs_epochs"]):
                cache.logger.info(
                    "[STAGE MARLFS] GPU %d client=%d epoch=%d/%d subset=%d micro_f1=%.6f macro_f1=%.6f train_loss=%.6f q_loss=%.6f candidate_seconds=%.2f",
                    cache.gpu_id,
                    client_id,
                    epoch,
                    int(config["marlfs_epochs"]),
                    len(selected),
                    performance,
                    float(evaluation["macro_f1"]),
                    float(evaluation["train_loss"]),
                    float(q_loss.detach().cpu()),
                    float(evaluation["seconds"]),
                )
            state = next_state
        cache.logger.info(
            "[STAGE MARLFS] GPU %d client=%d complete records=%d seconds=%.2f",
            cache.gpu_id,
            client_id,
            int(config["marlfs_epochs"]),
            time.perf_counter() - client_started,
        )
    atomic_json(output_path, records)
    return {
        "records_path": str(output_path),
        "record_count": len(records),
        "seconds": time.perf_counter() - started,
        "cache": cache.telemetry(),
    }


def pad_feature_sets(feature_sets: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(features) for features in feature_sets)
    tokens = torch.zeros((len(feature_sets), maximum), dtype=torch.long, device=device)
    membership = torch.zeros((len(feature_sets), 25), dtype=torch.float32, device=device)
    for row_id, features in enumerate(feature_sets):
        shifted = torch.tensor([feature_id + 1 for feature_id in features], dtype=torch.long, device=device)
        tokens[row_id, : len(features)] = shifted
        membership[row_id, torch.tensor(features, dtype=torch.long, device=device)] = 1.0
    padding_mask = tokens.eq(0)
    return tokens, padding_mask, membership


def train_set_autoencoder(records_path: Path, output_path: Path, config: dict, device: torch.device, logger: logging.Logger) -> dict:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    feature_sets = [list(map(int, record["feature_ids"])) for record in records]
    rng = np.random.default_rng(seed_from(int(config["seed"]), "encoder_split"))
    order = rng.permutation(len(feature_sets))
    validation_size = max(1, int(round(len(order) * 0.10)))
    validation_ids = order[:validation_size]
    train_ids = order[validation_size:]
    set_all_seeds(seed_from(int(config["seed"]), "set_autoencoder"))
    encoder = SetEncoder().to(device)
    decoder = SetDecoder().to(device)
    optimizer = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=float(config["encoder_learning_rate"]),
    )
    scaler = GradScaler("cuda")
    history = []
    best_loss = float("inf")
    best_states = None
    stale = 0
    started = time.perf_counter()
    logger.info(
        "[STAGE encoder] Start augmented_records=%s train=%s validation=%s max_epochs=%d patience=%d",
        f"{len(feature_sets):,}",
        f"{len(train_ids):,}",
        f"{len(validation_ids):,}",
        int(config["encoder_max_epochs"]),
        int(config["encoder_patience"]),
    )
    for epoch in range(1, int(config["encoder_max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        rng = np.random.default_rng(seed_from(int(config["seed"]), "encoder_epoch", epoch))
        shuffled = rng.permutation(train_ids)
        train_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        train_elements = 0
        encoder.train()
        decoder.train()
        for start in range(0, len(shuffled), int(config["encoder_batch_size"])):
            batch_ids = shuffled[start:start + int(config["encoder_batch_size"])]
            batch_sets = [feature_sets[int(index)] for index in batch_ids]
            tokens, padding_mask, membership = pad_feature_sets(batch_sets, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits, _ = decoder(encoder(tokens, padding_mask), padding_mask)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, membership)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.detach().double() * len(batch_sets)
            train_elements += len(batch_sets)
        encoder.eval()
        decoder.eval()
        validation_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        validation_elements = 0
        exact_sets = 0
        with torch.no_grad():
            for start in range(0, len(validation_ids), int(config["encoder_batch_size"])):
                batch_ids = validation_ids[start:start + int(config["encoder_batch_size"])]
                batch_sets = [feature_sets[int(index)] for index in batch_ids]
                tokens, padding_mask, membership = pad_feature_sets(batch_sets, device)
                with autocast("cuda"):
                    logits, _ = decoder(encoder(tokens, padding_mask), padding_mask)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, membership)
                predicted = logits.sigmoid().ge(0.5).to(membership.dtype)
                exact_sets += int(predicted.eq(membership).all(dim=1).sum().detach().cpu())
                validation_loss_sum += loss.detach().double() * len(batch_sets)
                validation_elements += len(batch_sets)
        train_loss = float(train_loss_sum.cpu()) / max(1, train_elements)
        validation_loss = float(validation_loss_sum.cpu()) / max(1, validation_elements)
        history.append({
            "epoch": epoch,
            "train_reconstruction_nll": train_loss,
            "validation_reconstruction_nll": validation_loss,
            "validation_exact_set_accuracy": exact_sets / max(1, validation_elements),
            "seconds": time.perf_counter() - epoch_started,
        })
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_states = {"encoder": cpu_state(encoder), "decoder": cpu_state(decoder), "epoch": epoch}
            stale = 0
        else:
            stale += 1
        logger.info(
            "[STAGE encoder] epoch=%d/%d train_nll=%.6f validation_nll=%.6f exact_set_accuracy=%.6f stale=%d seconds=%.2f",
            epoch,
            int(config["encoder_max_epochs"]),
            train_loss,
            validation_loss,
            exact_sets / max(1, validation_elements),
            stale,
            time.perf_counter() - epoch_started,
        )
        if stale >= int(config["encoder_patience"]):
            break
    if best_states is None:
        raise RuntimeError("Encoder-decoder did not produce a checkpoint")
    encoder.load_state_dict(best_states["encoder"])
    decoder.load_state_dict(best_states["decoder"])
    encoder.eval()
    decoder.eval()
    probe = feature_sets[int(validation_ids[0])]
    permuted = list(reversed(probe))
    with torch.no_grad():
        tokens_a, padding_a, _ = pad_feature_sets([probe], device)
        tokens_b, padding_b, _ = pad_feature_sets([permuted], device)
        logits_a, _ = decoder(encoder(tokens_a, padding_a), padding_a)
        logits_b, _ = decoder(encoder(tokens_b, padding_b), padding_b)
    invariance_error = float((logits_a - logits_b).abs().max().detach().cpu())
    if invariance_error > float(config["permutation_invariance_tolerance"]):
        raise AssertionError(f"Permutation invariance error {invariance_error}")
    torch.save({
        "encoder_state_dict": best_states["encoder"],
        "decoder_state_dict": best_states["decoder"],
        "best_epoch": int(best_states["epoch"]),
        "best_validation_loss": best_loss,
        "permutation_invariance_max_abs_error": invariance_error,
    }, output_path)
    logger.info(
        "[STAGE encoder] Complete best_epoch=%d best_validation_nll=%.6f invariance_error=%.9f seconds=%.2f",
        int(best_states["epoch"]),
        best_loss,
        invariance_error,
        time.perf_counter() - started,
    )
    return {
        "checkpoint_path": str(output_path),
        "history": history,
        "best_epoch": int(best_states["epoch"]),
        "best_validation_loss": best_loss,
        "permutation_invariance_max_abs_error": invariance_error,
        "seconds": time.perf_counter() - started,
    }


def train_final_clients(
    cache: GPUClientCache,
    client_ids: list[int],
    feature_ids_list: list[int],
    states: dict,
    round_id: int,
    config: dict,
    output_path: Path,
) -> dict:
    """Run exactly one local epoch and preserve each personalized state for the next round."""
    histories = []
    updated_states = {}
    started = time.perf_counter()
    for client_id in client_ids:
        family = config["client_models"][str(client_id)]
        client_started = time.perf_counter()
        entry = cache.get(client_id)
        model = cache.get_model_pool(family, 1)[0]
        model.load_state_dict(states[str(client_id)])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config["classifier_learning_rate"]),
            foreach=True,
        )
        scaler = GradScaler("cuda")
        train_examples = int(entry["meta"]["train_rows"])
        round_seed = seed_from(int(config["seed"]), "fedcaps_final_round", round_id, client_id)
        context = make_training_context(cache.device, train_examples, round_seed)
        stream = context["stream"]
        loss_sums = context["loss_sums"]
        if entry["mode"] == "gpu_full":
            training_order = context["order"]
            index_device = cache.device
        else:
            cpu_generator = torch.Generator().manual_seed(round_seed)
            training_order = torch.randperm(train_examples, generator=cpu_generator)
            index_device = torch.device("cpu")
        optimizer_steps = 0
        singleton_batches = 0
        with torch.cuda.stream(stream):
            feature_ids = torch.tensor(feature_ids_list, dtype=torch.long, device=cache.device)
            model.train()
            for start in range(0, train_examples, int(config["per_client_batch_size"])):
                end = min(train_examples, start + int(config["per_client_batch_size"]))
                positions = training_order[start:end]
                x, y = batch_from_entry(entry, "train", positions, feature_ids, cache.device)
                singleton = family == "cnn1d" and len(feature_ids_list) < 4 and x.shape[0] == 1
                if singleton:
                    bn_singleton_mode(model, True)
                    singleton_batches += 1
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sums[0] += loss.detach().double() * x.shape[0]
                loss_sums[1] += x.shape[0]
                optimizer_steps += 1
                if singleton:
                    bn_singleton_mode(model, False)
        stream.synchronize()
        train_loss = float(loss_sums[0].cpu()) / max(1.0, float(loss_sums[1].cpu()))
        histories.append({
            "round": round_id,
            "client_id": client_id,
            "model_family": family,
            "local_epoch": 1,
            "train_examples": train_examples,
            "validation_examples": 0,
            "optimizer_steps": optimizer_steps,
            "sampler_padding_rows": 0,
            "bn_singleton_fallback_batches": singleton_batches,
            "hard_loss": train_loss,
            "soft_loss": 0.0,
            "proximal_loss": 0.0,
            "total_loss": train_loss,
            "epoch_seconds": time.perf_counter() - client_started,
            "gpu_id": cache.gpu_id,
            "stream_count": 1,
            "cache_mode": entry["mode"],
        })
        updated_states[str(client_id)] = cpu_state(model)
        cache.logger.info(
            "[STAGE final-train] GPU %d round=%d client=%d steps=%d train_rows=%d loss=%.6f seconds=%.2f",
            cache.gpu_id, round_id, client_id, optimizer_steps, train_examples, train_loss,
            time.perf_counter() - client_started,
        )
    torch.save({"states": updated_states, "history_local_epoch": histories}, output_path)
    return {
        "payload_path": str(output_path),
        "client_count": len(client_ids),
        "history_rows": len(histories),
        "seconds": time.perf_counter() - started,
        "cache": cache.telemetry(),
    }


def test_final_clients(
    cache: GPUClientCache,
    client_ids: list[int],
    feature_ids_list: list[int],
    states: dict,
    global_test_meta: dict,
    config: dict,
    output_path: Path,
) -> dict:
    rows = int(global_test_meta["rows"])
    test_entry = cache.get_global_test(global_test_meta)
    results = []
    started = time.perf_counter()
    for client_id in client_ids:
        family = config["client_models"][str(client_id)]
        client_started = time.perf_counter()
        cache.logger.info(
            "[STAGE final-test] GPU %d client=%d family=%s start test_rows=%s selected_features=%d",
            cache.gpu_id,
            client_id,
            family,
            f"{rows:,}",
            len(feature_ids_list),
        )
        model = cache.get_model_pool(family, 1)[0]
        model.load_state_dict(states[str(client_id)])
        stream = torch.cuda.Stream(device=cache.device)
        confusion, loss_sum, examples = evaluate_model(
            model,
            test_entry,
            "test",
            feature_ids_list,
            int(config["per_client_batch_size"]),
            cache.device,
            stream,
        )
        metrics = metrics_from_confusion(confusion, loss_sum, examples)
        results.append({
            "client_id": client_id,
            "model_family": family,
            "metrics": metrics,
            "confusion_matrix": confusion.tolist(),
            "gpu_id": cache.gpu_id,
        })
        cache.logger.info(
            "[STAGE final-test] GPU %d client=%d complete accuracy=%.6f macro_f1=%.6f weighted_f1=%.6f seconds=%.2f",
            cache.gpu_id,
            client_id,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
            time.perf_counter() - client_started,
        )
    atomic_json(output_path, results)
    return {
        "payload_path": str(output_path),
        "client_count": len(client_ids),
        "seconds": time.perf_counter() - started,
        "cache": cache.telemetry(),
    }


def gpu_utilization_sample(gpu_id: int) -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        return {
            "gpu_id": gpu_id,
            "utilization_percent": float(values[0]),
            "memory_used_mib": float(values[1]),
            "memory_total_mib": float(values[2]),
            "temperature_c": float(values[3]),
            "power_w": float(values[4]),
            "timestamp": time.time(),
        }
    except Exception as error:
        return {"gpu_id": gpu_id, "error": str(error), "timestamp": time.time()}


class TaskGPUMonitor:
    def __init__(self, gpu_id: int, task_id: str, kind: str, interval_seconds: float) -> None:
        self.gpu_id = gpu_id
        self.task_id = task_id
        self.kind = kind
        self.interval_seconds = interval_seconds
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"gpu-monitor-{gpu_id}", daemon=True)

    def _sample(self) -> None:
        self.samples.append(gpu_utilization_sample(self.gpu_id) | {
            "task_id": self.task_id,
            "task_kind": self.kind,
        })

    def _run(self) -> None:
        self._sample()
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[dict]:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, 2.0 * self.interval_seconds))
        self._sample()
        return self.samples


def worker_main(gpu_id, task_queue, result_queue, manifest_path):
    logger = None
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        config = manifest["config"]
        paths = {name: Path(value) for name, value in manifest["paths"].items()}
        log_path = paths["logs"] / f"worker_{gpu_id}.log"
        logger = setup_logger(log_path, f"fedcaps-worker-{gpu_id}")
        logger.info("Worker %d process started; binding CUDA device next", gpu_id)
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
        logger.info("Worker %d started on %s", gpu_id, torch.cuda.get_device_name(gpu_id))
        cache = None
        initial_states = torch.load(paths["temporary"] / "initial_states.pt", map_location="cpu", weights_only=False)
        while True:
            task = task_queue.get()
            task_id = str(task["task_id"])
            kind = str(task["kind"])
            if kind == "shutdown":
                result_queue.put({"kind": "shutdown_ok", "task_id": task_id, "gpu_id": gpu_id})
                break
            monitor = None
            try:
                task_started = time.perf_counter()
                logger.info("[TASK] GPU %d task=%s kind=%s start", gpu_id, task_id, kind)
                monitor = TaskGPUMonitor(
                    gpu_id,
                    task_id,
                    kind,
                    float(config["gpu_monitor_interval_seconds"]),
                )
                monitor.start()
                if kind == "benchmark":
                    payload = {
                        family: [benchmark_family(family, config, device, stream_count) for stream_count in config["stream_candidates"]]
                        for family in task["families"]
                    }
                elif kind == "initialize_cache":
                    config["runtime_stream_count_by_gpu"] = task["runtime_stream_count_by_gpu"]
                    if cache is not None:
                        logger.info(
                            "[STAGE cache] GPU %d releasing phase=%s clients=%s before rebalance",
                            gpu_id,
                            cache.phase,
                            cache.owned_client_ids,
                        )
                        del cache
                        cache = None
                        torch.cuda.empty_cache()
                    cache = GPUClientCache(
                        gpu_id,
                        config,
                        manifest["dataset_summary"]["clients"],
                        logger,
                        str(task["phase"]),
                    )
                    payload = cache.initialize([int(value) for value in task["client_ids"]])
                elif kind == "collect_records":
                    if cache is None:
                        raise RuntimeError("Cache has not been initialized")
                    output_path = paths["temporary"] / f"worker_{gpu_id}_records.json"
                    payload = collect_marlfs_records(cache, [int(value) for value in task["client_ids"]], initial_states, config, output_path)
                elif kind == "train_autoencoder":
                    output_path = paths["temporary"] / "set_autoencoder.pt"
                    payload = train_set_autoencoder(Path(task["records_path"]), output_path, config, device, logger)
                elif kind == "evaluate_subsets":
                    if cache is None:
                        raise RuntimeError("Cache has not been initialized")
                    evaluations = []
                    for candidate in task["candidates"]:
                        for client_id in task["client_ids"]:
                            family = config["client_models"][str(client_id)]
                            evaluations.append(evaluate_candidate(
                                cache,
                                int(client_id),
                                family,
                                [int(value) for value in candidate["feature_ids"]],
                                initial_states[family],
                                config,
                                seed_from(int(config["seed"]), "ppo_feedback", int(candidate["candidate_id"]), int(client_id)),
                            ) | {"candidate_id": int(candidate["candidate_id"])})
                    output_path = paths["temporary"] / f"worker_{gpu_id}_candidate_{task_id}.json"
                    atomic_json(output_path, evaluations)
                    payload = {"evaluations_path": str(output_path), "evaluation_count": len(evaluations), "cache": cache.telemetry()}
                elif kind == "train_final":
                    if cache is None:
                        raise RuntimeError("Cache has not been initialized")
                    output_path = paths["temporary"] / f"worker_{gpu_id}_round_{int(task['round']):03d}_training.pt"
                    payload = train_final_clients(
                        cache,
                        [int(value) for value in task["client_ids"]],
                        [int(value) for value in task["feature_ids"]],
                        task["states"],
                        int(task["round"]),
                        config,
                        output_path,
                    )
                elif kind == "test_final":
                    if cache is None:
                        raise RuntimeError("Cache has not been initialized")
                    output_path = paths["temporary"] / f"worker_{gpu_id}_round_{int(task['round']):03d}_test.json"
                    payload = test_final_clients(
                        cache,
                        [int(value) for value in task["client_ids"]],
                        [int(value) for value in task["feature_ids"]],
                        task["states"],
                        manifest["dataset_summary"]["global_test"],
                        config,
                        output_path,
                    )
                else:
                    raise ValueError(f"Unknown task kind: {kind}")
                logger.info(
                    "[TASK] GPU %d task=%s kind=%s complete seconds=%.2f",
                    gpu_id,
                    task_id,
                    kind,
                    time.perf_counter() - task_started,
                )
                task_utilization_samples = monitor.stop()
                monitor = None
                result_queue.put({
                    "kind": "task_ok",
                    "task_id": task_id,
                    "gpu_id": gpu_id,
                    "payload": payload,
                    "utilization_samples": task_utilization_samples,
                })
            except Exception:
                if monitor is not None:
                    monitor.stop()
                result_queue.put({
                    "kind": "worker_error",
                    "task_id": task_id,
                    "gpu_id": gpu_id,
                    "traceback": traceback.format_exc(),
                })
                raise
    except Exception:
        if logger is not None:
            logger.exception("Worker %d failed", gpu_id)
        try:
            result_queue.put({"kind": "worker_error", "gpu_id": gpu_id, "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise


def wait_for_results(result_queue, expected_task_ids: set[str], workers, timeout_seconds: int) -> list[dict]:
    received = {}
    deadline = time.monotonic() + timeout_seconds
    while set(received) != expected_task_ids:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for tasks {sorted(expected_task_ids - set(received))}")
        try:
            result = result_queue.get(timeout=min(30.0, remaining))
        except queue.Empty:
            failed = [(index, worker.exitcode) for index, worker in enumerate(workers) if worker.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker exited before result: {failed}")
            continue
        if result.get("kind") == "worker_error":
            raise RuntimeError(f"Worker error on GPU {result.get('gpu_id')}:\n{result.get('traceback')}")
        task_id = str(result.get("task_id"))
        if task_id not in expected_task_ids:
            raise RuntimeError(f"Unexpected task result: {task_id}")
        if task_id in received:
            raise RuntimeError(f"Duplicate task result: {task_id}")
        received[task_id] = result
    return [received[task_id] for task_id in sorted(expected_task_ids)]


def dispatch(task_queues, result_queue, workers, tasks: list[tuple[int, dict]], timeout_seconds: int) -> list[dict]:
    expected = set()
    for gpu_id, task in tasks:
        task_id = str(task["task_id"])
        if task_id in expected:
            raise ValueError(f"Duplicate task ID before dispatch: {task_id}")
        expected.add(task_id)
        task_queues[gpu_id].put(task)
    return wait_for_results(result_queue, expected, workers, timeout_seconds)


def optimize_bipartition(work_units: dict[int, float], seconds_per_unit: dict[int, dict[str, float]], config: dict) -> tuple[dict, dict]:
    clients = list(range(1, int(config["num_clients"]) + 1))
    best = None
    for mask in range(1, 2 ** len(clients) - 1):
        groups = (
            tuple(client_id for index, client_id in enumerate(clients) if mask & (1 << index)),
            tuple(client_id for index, client_id in enumerate(clients) if not mask & (1 << index)),
        )
        loads = []
        for gpu_id, group in enumerate(groups):
            load = 0.0
            for client_id in group:
                family = config["client_models"][str(client_id)]
                load += float(work_units[client_id]) * float(seconds_per_unit[gpu_id][family])
            loads.append(load)
        key = (max(loads), abs(loads[0] - loads[1]), groups[0], groups[1])
        if best is None or key < best[0]:
            best = (key, groups, loads)
    if best is None:
        raise RuntimeError("No nontrivial client bipartition")
    assignment = {"0": list(best[1][0]), "1": list(best[1][1])}
    return assignment, {
        "predicted_load_seconds": {"0": best[2][0], "1": best[2][1]},
        "predicted_makespan_seconds": best[0][0],
        "predicted_idle_seconds": {
            "0": best[0][0] - best[2][0],
            "1": best[0][0] - best[2][1],
        },
        "work_units": {str(client_id): float(work_units[client_id]) for client_id in clients},
    }


def select_client_assignments(dataset_summary: dict, benchmark_results: dict, config: dict) -> tuple[dict, dict]:
    single_stream_seconds = {}
    search_seconds = {}
    stream_selection = {}
    benchmark_recommendation = {}
    stream_selection_reason = {}
    for gpu_id in range(2):
        single_stream_seconds[gpu_id] = {}
        search_seconds[gpu_id] = {}
        stream_selection[str(gpu_id)] = {}
        benchmark_recommendation[str(gpu_id)] = {}
        stream_selection_reason[str(gpu_id)] = {}
        for family, candidates in benchmark_results[str(gpu_id)].items():
            candidates = sorted(candidates, key=lambda row: int(row["stream_count"]))
            one = candidates[0]
            two = candidates[-1]
            throughput_recommends_two = (
                float(two["samples_per_second"])
                > float(one["samples_per_second"]) * float(config["stream_improvement_margin"])
            )
            benchmark_recommendation[str(gpu_id)][family] = 2 if throughput_recommends_two else 1
            replay_is_acceptable = (
                bool(two["deterministic_replay_valid"])
                or not bool(config["require_multistream_deterministic_replay"])
            )
            safe_two_streams = throughput_recommends_two and replay_is_acceptable
            selected = two if safe_two_streams else one
            stream_selection[str(gpu_id)][family] = int(selected["stream_count"])
            single_stream_seconds[gpu_id][family] = float(one["seconds_per_step"])
            search_seconds[gpu_id][family] = float(selected["seconds_per_step"])
            if safe_two_streams:
                reason = "two_streams_measured_faster_and_deterministic_replay_matched"
            elif throughput_recommends_two:
                reason = "single_stream_selected_because_two_stream_deterministic_replay_differed"
            else:
                reason = "single_stream_measured_faster_or_equal"
            stream_selection_reason[str(gpu_id)][family] = reason

    batch_size = int(config["per_client_batch_size"])
    feedback_candidates = (
        math.ceil(int(config["ppo_steps_per_epoch"]) / int(config["ppo_feedback_interval"]))
        * int(config["ppo_search_epochs"])
    )
    search_candidate_count = int(config["marlfs_epochs"]) + feedback_candidates
    search_work = {}
    final_training_work = {}
    evaluation_work = {}
    global_test_steps = math.ceil(int(dataset_summary["global_test"]["rows"]) / batch_size)
    for client_id in range(1, int(config["num_clients"]) + 1):
        meta = dataset_summary["clients"][str(client_id)]
        cv_steps = 0
        for fold_id in range(int(config["fedcaps_search_cv_folds"])):
            cv_steps += math.ceil(int(meta[f"search_fold_{fold_id}_train_rows"]) / batch_size)
            cv_steps += math.ceil(int(meta[f"search_fold_{fold_id}_validation_rows"]) / batch_size)
        search_work[client_id] = cv_steps * search_candidate_count
        final_training_work[client_id] = math.ceil(int(meta["train_rows"]) / batch_size) * int(config["rounds"])
        evaluation_work[client_id] = global_test_steps * int(config["rounds"])

    search_assignment, search_details = optimize_bipartition(search_work, search_seconds, config)
    final_training_assignment, final_training_details = optimize_bipartition(
        final_training_work,
        single_stream_seconds,
        config,
    )
    evaluation_assignment, evaluation_details = optimize_bipartition(
        evaluation_work,
        single_stream_seconds,
        config,
    )
    assignments = {
        "search": search_assignment,
        "final_training": final_training_assignment,
        "evaluation": evaluation_assignment,
    }
    details = {
        "phase_assignments": assignments,
        "predicted_by_phase": {
            "search": search_details,
            "final_training": final_training_details,
            "evaluation": evaluation_details,
        },
        "search_candidate_count_per_client": search_candidate_count,
        "stream_selection_by_gpu_and_family": stream_selection,
        "stream_benchmark_recommendation_by_gpu_and_family": benchmark_recommendation,
        "stream_selection_reason_by_gpu_and_family": stream_selection_reason,
    }
    config["runtime_stream_count_by_gpu"] = {
        str(gpu_id): max(stream_selection[str(gpu_id)].values()) for gpu_id in range(2)
    }
    config["client_assignments_by_phase"] = assignments
    return assignments, details


def augment_records(original_records: list[dict], permutations: int, seed: int) -> list[dict]:
    augmented = []
    for record in original_records:
        features = [int(value) for value in record["feature_ids"]]
        for permutation_id in range(permutations):
            rng = np.random.default_rng(seed_from(seed, "record_permutation", permutation_id, int(record["client_id"])) + int(record["collection_epoch"]))
            permuted = np.asarray(features)[rng.permutation(len(features))].astype(int).tolist()
            augmented.append({
                **record,
                "source_record_id": record["record_id"],
                "record_id": f"{record['record_id']}_p{permutation_id + 1:02d}",
                "permutation_id": permutation_id + 1,
                "feature_ids": permuted,
                "feature_names": [FEATURE_COLUMNS[feature_id] for feature_id in permuted],
            })
    return augmented


def records_for_csv(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append({
            **record,
            "feature_ids": json.dumps(record["feature_ids"], default=json_compatible),
            "feature_names": json.dumps(record["feature_names"], ensure_ascii=False, default=json_compatible),
        })
    return rows


def sample_weighted_candidate(evaluations: list[dict], dataset_summary: dict) -> dict:
    total_train = sum(int(dataset_summary["clients"][str(client_id)]["train_rows"]) for client_id in range(1, 11))
    weighted_micro = 0.0
    weighted_macro = 0.0
    per_client = []
    for evaluation in sorted(evaluations, key=lambda row: int(row["client_id"])):
        train_rows = int(dataset_summary["clients"][str(evaluation["client_id"])]["train_rows"])
        weight = train_rows / total_train
        weighted_micro += weight * float(evaluation["micro_f1"])
        weighted_macro += weight * float(evaluation["macro_f1"])
        per_client.append({**evaluation, "sample_weight": weight})
    return {"weighted_micro_f1": weighted_micro, "weighted_macro_f1": weighted_macro, "per_client": per_client}


def masks_from_records(records: list[dict]) -> torch.Tensor:
    masks = torch.zeros((len(records), 25), dtype=torch.float32)
    for row_id, record in enumerate(records):
        masks[row_id, torch.tensor(record["feature_ids"], dtype=torch.long)] = 1.0
    return masks


def decode_latent_masks(decoder: SetDecoder, latent: torch.Tensor, minimum_size: int) -> tuple[torch.Tensor, list[list[int]]]:
    logits = decoder.logits_from_latent(latent)
    masks = logits.sigmoid().ge(0.5).to(torch.float32)
    feature_sets = []
    for row_id in range(masks.shape[0]):
        selected = torch.nonzero(masks[row_id] > 0, as_tuple=False).flatten()
        if selected.numel() < minimum_size:
            selected = torch.topk(logits[row_id], k=minimum_size).indices
            masks[row_id].zero_()
            masks[row_id, selected] = 1.0
        feature_sets.append(sorted(int(value) for value in selected.tolist()))
    return masks, feature_sets


def encode_feature_sets(encoder: SetEncoder, decoder: SetDecoder, feature_sets: list[list[int]]) -> torch.Tensor:
    tokens, padding_mask, _ = pad_feature_sets(feature_sets, torch.device("cpu"))
    with torch.no_grad():
        embedding = encoder(tokens, padding_mask)
        _, latent = decoder(embedding, padding_mask)
    return latent


def train_initial_reward_critic(critic: RewardCritic, records: list[dict], dataset_summary: dict, config: dict) -> None:
    masks = masks_from_records(records)
    targets = torch.tensor([float(record["performance_micro_f1"]) for record in records], dtype=torch.float32)
    total_train = sum(int(dataset_summary["clients"][str(client_id)]["train_rows"]) for client_id in range(1, 11))
    weights = torch.tensor([
        int(dataset_summary["clients"][str(record["client_id"])]["train_rows"]) / total_train
        for record in records
    ], dtype=torch.float32)
    weights = weights / weights.mean()
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["critic_learning_rate"]))
    generator = torch.Generator().manual_seed(seed_from(int(config["seed"]), "critic_pretrain"))
    for epoch in range(int(config["critic_pretrain_epochs"])):
        order = torch.randperm(len(records), generator=generator)
        for start in range(0, len(order), int(config["ppo_batch_size"])):
            ids = order[start:start + int(config["ppo_batch_size"])]
            prediction = critic(masks[ids])
            loss = ((prediction - targets[ids]).square() * weights[ids]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def run_ppo_search(
    config: dict,
    dataset_summary: dict,
    original_records: list[dict],
    autoencoder_path: Path,
    task_queues,
    result_queue,
    workers,
    assignment: dict,
    logger: logging.Logger,
) -> dict:
    checkpoint = torch.load(autoencoder_path, map_location="cpu", weights_only=False)
    encoder = SetEncoder()
    decoder = SetDecoder()
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    ranked = sorted(original_records, key=lambda row: (-float(row["performance_micro_f1"]), len(row["feature_ids"]), row["record_id"]))
    seeds = ranked[: int(config["ppo_seed_count"])]
    seed_feature_sets = [list(map(int, record["feature_ids"])) for record in seeds]
    seed_latents = encode_feature_sets(encoder, decoder, seed_feature_sets)
    seed_masks = masks_from_records(seeds)
    actor = GaussianActor()
    reward_critic = RewardCritic()
    value_critic = RewardCritic()
    train_initial_reward_critic(reward_critic, original_records, dataset_summary, config)
    value_critic.load_state_dict(reward_critic.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["actor_learning_rate"]))
    value_optimizer = torch.optim.Adam(value_critic.parameters(), lr=float(config["critic_learning_rate"]))
    reward_optimizer = torch.optim.Adam(reward_critic.parameters(), lr=float(config["critic_learning_rate"]))
    generator = torch.Generator().manual_seed(seed_from(int(config["seed"]), "ppo"))
    history = []
    candidate_rows = []
    seen = set()
    candidate_id = 0
    global_step = 0
    communication_cumulative = 0
    started = time.perf_counter()
    logger.info(
        "[STAGE PPO] Start seeds=%d epochs=%d steps_per_epoch=%d feedback_interval=%d batch=%d",
        len(seeds),
        int(config["ppo_search_epochs"]),
        int(config["ppo_steps_per_epoch"]),
        int(config["ppo_feedback_interval"]),
        int(config["ppo_batch_size"]),
    )
    for search_epoch in range(1, int(config["ppo_search_epochs"]) + 1):
        epoch_started = time.perf_counter()
        rollout_latents = []
        rollout_actions = []
        rollout_old_logprob = []
        rollout_seed_masks = []
        rollout_candidate_masks = []
        rollout_rewards = []
        epoch_candidates = []
        for chunk_start in range(0, int(config["ppo_steps_per_epoch"]), int(config["ppo_feedback_interval"])):
            chunk_size = min(int(config["ppo_feedback_interval"]), int(config["ppo_steps_per_epoch"]) - chunk_start)
            seed_ids = torch.randint(0, len(seeds), (chunk_size,), generator=generator)
            base_latent = seed_latents[seed_ids]
            distribution = actor.distribution(base_latent)
            action = distribution.sample()
            old_logprob = distribution.log_prob(action).sum(dim=1).detach()
            enhanced = base_latent + float(config["actor_step_scale"]) * action
            candidate_masks, feature_sets = decode_latent_masks(decoder, enhanced.detach(), int(config["minimum_subset_size"]))
            with torch.no_grad():
                predicted_performance = reward_critic(candidate_masks).clamp(0.0, 1.0)
                baseline_performance = reward_critic(seed_masks[seed_ids]).clamp(0.0, 1.0)
                compactness = 1.0 - candidate_masks.mean(dim=1)
                predicted_reward = float(config["reward_lambda"]) * (predicted_performance - baseline_performance) + (1.0 - float(config["reward_lambda"])) * compactness
            best_index = int(torch.argmax(predicted_reward).detach().cpu())
            proposed = feature_sets[best_index]
            proposed_key = tuple(proposed)
            if proposed_key in seen:
                unique_order = torch.argsort(predicted_reward, descending=True).tolist()
                for index in unique_order:
                    key = tuple(feature_sets[int(index)])
                    if key not in seen:
                        best_index = int(index)
                        proposed = feature_sets[best_index]
                        proposed_key = key
                        break
            seen.add(proposed_key)
            candidate_id += 1
            candidate = {"candidate_id": candidate_id, "feature_ids": proposed}
            tasks = [
                (
                    gpu_id,
                    {
                        "task_id": f"ppo_eval_e{search_epoch:02d}_c{candidate_id:03d}_g{gpu_id}",
                        "kind": "evaluate_subsets",
                        "client_ids": assignment[str(gpu_id)],
                        "candidates": [candidate],
                    },
                )
                for gpu_id in range(2)
            ]
            results = dispatch(task_queues, result_queue, workers, tasks, int(config["worker_timeout_seconds"]))
            evaluations = []
            for result in results:
                evaluations.extend(json.loads(Path(result["payload"]["evaluations_path"]).read_text(encoding="utf-8")))
            aggregated = sample_weighted_candidate(evaluations, dataset_summary)
            baseline_value = float(baseline_performance[best_index].detach().cpu())
            compactness_value = 1.0 - len(proposed) / 25.0
            actual_reward = float(config["reward_lambda"]) * (aggregated["weighted_micro_f1"] - baseline_value) + (1.0 - float(config["reward_lambda"])) * compactness_value
            predicted_reward[best_index] = actual_reward
            candidate_communication = 10 * (2 * len(proposed) + 4)
            communication_cumulative += candidate_communication
            row = {
                "candidate_id": candidate_id,
                "search_epoch": search_epoch,
                "global_step": global_step + chunk_size,
                "feature_ids": proposed,
                "feature_names": [FEATURE_COLUMNS[feature_id] for feature_id in proposed],
                "subset_size": len(proposed),
                "weighted_micro_f1": aggregated["weighted_micro_f1"],
                "weighted_macro_f1": aggregated["weighted_macro_f1"],
                "baseline_performance": baseline_value,
                "compactness": compactness_value,
                "reward": actual_reward,
                "communication_bytes": candidate_communication,
                "communication_bytes_cumulative": communication_cumulative,
                "per_client": aggregated["per_client"],
            }
            candidate_rows.append(row)
            epoch_candidates.append(row)
            logger.info(
                "[STAGE PPO] calibration candidate=%d epoch=%d step=%d subset=%d weighted_micro_f1=%.6f weighted_macro_f1=%.6f reward=%.6f communication_bytes=%d",
                candidate_id,
                search_epoch,
                global_step + chunk_size,
                len(proposed),
                aggregated["weighted_micro_f1"],
                aggregated["weighted_macro_f1"],
                actual_reward,
                candidate_communication,
            )
            calibration_mask = candidate_masks[best_index:best_index + 1]
            calibration_target = torch.tensor([aggregated["weighted_micro_f1"]], dtype=torch.float32)
            for calibration_step in range(int(config["critic_calibration_steps"])):
                reward_loss = nn.functional.mse_loss(reward_critic(calibration_mask), calibration_target)
                reward_optimizer.zero_grad(set_to_none=True)
                reward_loss.backward()
                reward_optimizer.step()
            rollout_latents.append(base_latent)
            rollout_actions.append(action.detach())
            rollout_old_logprob.append(old_logprob)
            rollout_seed_masks.append(seed_masks[seed_ids])
            rollout_candidate_masks.append(candidate_masks)
            rollout_rewards.append(predicted_reward.detach())
            global_step += chunk_size
        latent_batch = torch.cat(rollout_latents)
        action_batch = torch.cat(rollout_actions)
        old_logprob_batch = torch.cat(rollout_old_logprob)
        seed_mask_batch = torch.cat(rollout_seed_masks)
        candidate_mask_batch = torch.cat(rollout_candidate_masks)
        reward_batch = torch.cat(rollout_rewards)
        with torch.no_grad():
            value_before = value_critic(seed_mask_batch)
            advantages = reward_batch - value_before
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            returns = reward_batch
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        for update_epoch in range(int(config["ppo_update_epochs"])):
            order = torch.randperm(len(reward_batch), generator=generator)
            for start in range(0, len(order), int(config["ppo_batch_size"])):
                ids = order[start:start + int(config["ppo_batch_size"])]
                distribution = actor.distribution(latent_batch[ids])
                new_logprob = distribution.log_prob(action_batch[ids]).sum(dim=1)
                ratio = (new_logprob - old_logprob_batch[ids]).exp()
                unclipped = ratio * advantages[ids]
                clipped = ratio.clamp(1.0 - float(config["ppo_clip_ratio"]), 1.0 + float(config["ppo_clip_ratio"])) * advantages[ids]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                entropy_bonus = distribution.entropy().sum(dim=1).mean()
                total_actor_loss = actor_loss - float(config["ppo_entropy_coefficient"]) * entropy_bonus
                actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                actor_optimizer.step()
                value_prediction = value_critic(seed_mask_batch[ids])
                critic_loss = nn.functional.mse_loss(value_prediction, returns[ids])
                value_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                value_optimizer.step()
                actor_loss_value = float(actor_loss.detach())
                critic_loss_value = float(critic_loss.detach())
        best_epoch_candidate = sorted(epoch_candidates, key=lambda row: (-float(row["reward"]), -float(row["weighted_micro_f1"]), int(row["subset_size"]), tuple(row["feature_ids"])))[0]
        history.append({
            "search_epoch": search_epoch,
            "round": search_epoch,
            "rollout_steps": int(config["ppo_steps_per_epoch"]),
            "feedback_candidates": len(epoch_candidates),
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "mean_predicted_reward": float(reward_batch.mean()),
            "best_reward": float(best_epoch_candidate["reward"]),
            "best_weighted_micro_f1": float(best_epoch_candidate["weighted_micro_f1"]),
            "best_weighted_macro_f1": float(best_epoch_candidate["weighted_macro_f1"]),
            "best_subset_size": int(best_epoch_candidate["subset_size"]),
            "best_feature_ids": best_epoch_candidate["feature_ids"],
            "round_seconds": time.perf_counter() - epoch_started,
            "communication_bytes_round": sum(int(row["communication_bytes"]) for row in epoch_candidates),
            "communication_bytes_cumulative": communication_cumulative,
        })
        logger.info("PPO epoch %d best reward %.6f weighted Micro-F1 %.6f subset %d", search_epoch, best_epoch_candidate["reward"], best_epoch_candidate["weighted_micro_f1"], best_epoch_candidate["subset_size"])
    if not candidate_rows:
        raise RuntimeError("PPO produced no client-calibrated candidates")
    best_by_reward = sorted(candidate_rows, key=lambda row: (-float(row["reward"]), -float(row["weighted_micro_f1"]), int(row["subset_size"]), tuple(row["feature_ids"])))[0]
    best_by_performance = sorted(candidate_rows, key=lambda row: (-float(row["weighted_micro_f1"]), int(row["subset_size"]), tuple(row["feature_ids"])))[0]
    logger.info(
        "[STAGE PPO] Complete selected_subset=%d best_reward=%.6f weighted_micro_f1=%.6f candidates=%d seconds=%.2f",
        len(best_by_reward["feature_ids"]),
        best_by_reward["reward"],
        best_by_reward["weighted_micro_f1"],
        len(candidate_rows),
        time.perf_counter() - started,
    )
    return {
        "history": history,
        "candidate_evaluations": candidate_rows,
        "best_by_reward": best_by_reward,
        "best_by_weighted_micro_f1": best_by_performance,
        "selected_feature_ids": best_by_reward["feature_ids"],
        "actor_state_dict": cpu_state(actor),
        "critic_state_dict": cpu_state(value_critic),
        "reward_critic_state_dict": cpu_state(reward_critic),
        "seconds": time.perf_counter() - started,
        "communication_bytes": communication_cumulative,
    }


def weighted_metric_average(test_rows: list[dict], dataset_summary: dict, weighted: bool) -> dict:
    metric_names = [name for name in test_rows[0]["metrics"] if name != "examples"]
    if weighted:
        total = sum(int(dataset_summary["clients"][str(row["client_id"])]["train_rows"]) for row in test_rows)
        weights = [int(dataset_summary["clients"][str(row["client_id"])]["train_rows"]) / total for row in test_rows]
    else:
        weights = [1.0 / len(test_rows)] * len(test_rows)
    return {
        name: float(sum(weight * float(row["metrics"][name]) for weight, row in zip(weights, test_rows)))
        for name in metric_names
    }


def save_plots(
    paths: dict[str, Path],
    dataset_summary: dict,
    local_history: list[dict],
    ppo_history: list[dict],
    candidate_rows: list[dict],
    confusion: np.ndarray,
    report_rows: list[dict],
    selected_feature_ids: list[int],
    evaluation_rows: list[dict],
    round_rows: list[dict],
    utilization_rows: list[dict],
) -> float:
    started = time.perf_counter()
    artifact = paths["artifacts"]
    client_totals = [int(dataset_summary["clients"][str(client_id)]["rows"]) for client_id in range(1, 11)]
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, 11), client_totals)
    plt.xlabel("Client")
    plt.ylabel("Rows")
    plt.title("CICIoT2023 client distribution")
    plt.tight_layout()
    plt.savefig(artifact / "class_distribution.png", dpi=160)
    plt.close()
    frame = pd.DataFrame(local_history)
    evaluation = pd.DataFrame(evaluation_rows)
    curves = evaluation.groupby("round")[["accuracy", "macro_f1", "weighted_f1"]].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(curves.index, curves["accuracy"], label="Mean client accuracy")
    plt.plot(curves.index, curves["macro_f1"], label="Mean client macro-F1")
    plt.plot(curves.index, curves["weighted_f1"], label="Mean client weighted-F1")
    plt.legend()
    plt.xlabel("Personalized training round")
    plt.tight_layout()
    plt.savefig(artifact / "evaluation_metric_curves.png", dpi=160)
    plt.close()
    losses = frame.groupby("round")[["hard_loss"]].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(losses.index, losses["hard_loss"], label="Train CE")
    plt.legend()
    plt.xlabel("Personalized training round")
    plt.tight_layout()
    plt.savefig(artifact / "loss_curves.png", dpi=160)
    plt.close()
    normalized = np.divide(confusion, confusion.sum(axis=1, keepdims=True), out=np.zeros_like(confusion, dtype=np.float64), where=confusion.sum(axis=1, keepdims=True) > 0)
    plt.figure(figsize=(12, 10))
    plt.imshow(normalized, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Row-normalized rate")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(artifact / "confusion_matrix.png", dpi=180)
    plt.close()
    plt.figure(figsize=(12, 5))
    plt.bar(range(NUM_CLASSES), [float(row["f1"]) for row in report_rows])
    plt.xlabel("Class ID")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(artifact / "per_class_f1.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar([int(row["round"]) for row in round_rows], [float(row["round_seconds"]) for row in round_rows])
    plt.xlabel("Personalized training round")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(artifact / "runtime_per_round.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot([int(row["search_epoch"]) for row in ppo_history], [int(row["communication_bytes_cumulative"]) / (1024**2) for row in ppo_history], marker="o")
    plt.xlabel("PPO search epoch")
    plt.ylabel("Logical communication (MiB)")
    plt.tight_layout()
    plt.savefig(artifact / "communication_cumulative.png", dpi=160)
    plt.close()
    valid_utilization = [row for row in utilization_rows if "utilization_percent" in row]
    plt.figure(figsize=(9, 4))
    for gpu_id in range(2):
        rows = [row for row in valid_utilization if int(row["gpu_id"]) == gpu_id]
        plt.plot(range(len(rows)), [float(row["utilization_percent"]) for row in rows], marker="o", label=f"GPU {gpu_id}")
    plt.xlabel("Sample")
    plt.ylabel("GPU utilization (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact / "gpu_utilization.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 4))
    plt.scatter([int(row["subset_size"]) for row in candidate_rows], [float(row["weighted_micro_f1"]) for row in candidate_rows], c=[float(row["reward"]) for row in candidate_rows], cmap="viridis")
    plt.colorbar(label="FedCAPS reward")
    plt.xlabel("Subset size")
    plt.ylabel("Sample-weighted Micro-F1")
    plt.tight_layout()
    plt.savefig(artifact / "subset_size_search.png", dpi=160)
    plt.close()
    selection = np.zeros(25, dtype=np.float64)
    selection[selected_feature_ids] = 1.0
    plt.figure(figsize=(12, 5))
    plt.bar(range(25), selection)
    plt.xticks(range(25), FEATURE_COLUMNS, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Selected by final FedCAPS subset")
    plt.tight_layout()
    plt.savefig(artifact / "selected_feature_importance.png", dpi=180)
    plt.close()
    return time.perf_counter() - started


def required_output_paths(paths: dict[str, Path]) -> list[Path]:
    relative = [
        "checkpoints/checkpoint_manifest.json", "logs/run.log", "logs/worker_0.log", "logs/worker_1.log",
        "metrics/config.json", "metrics/dataset_summary.json", "metrics/client_class_distribution.csv",
        "metrics/history_pretrain.csv", "metrics/history_pretrain.json",
        "metrics/history_round.csv", "metrics/history_round.json", "metrics/history_client.csv",
        "metrics/history_client.json", "metrics/history_local_epoch.csv", "metrics/history_local_epoch.json",
        "metrics/evaluation_metrics.csv", "metrics/evaluation_metrics.json", "metrics/final_round_metrics.csv",
        "metrics/final_round_metrics.json", "metrics/server_evaluation_status.json", "metrics/summary.json", "metrics/classification_report.json",
        "metrics/classification_report.csv", "metrics/confusion_matrix.csv", "metrics/confusion_matrix.npy",
        "metrics/communication_costs.json", "metrics/communication_costs.csv", "metrics/runtime_breakdown.json",
        "artifacts/class_distribution.png", "artifacts/evaluation_metric_curves.png", "artifacts/loss_curves.png", "artifacts/confusion_matrix.png",
        "artifacts/per_class_f1.png", "artifacts/runtime_per_round.png", "artifacts/communication_cumulative.png",
        "artifacts/gpu_utilization.png", "metrics/feature_selection_records.csv", "metrics/encoder_history.csv",
        "metrics/ppo_history.csv", "metrics/candidate_evaluations.csv", "metrics/selected_features.json",
    ]
    return [paths["root"] / value for value in relative]


def verify_checkpoint_manifest(output_dir: Path, config: dict, rows: list[dict]) -> None:
    manifest = json.loads((output_dir / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != int(config["rounds"]):
        raise AssertionError("Checkpoint manifest must contain one entry per round")
    for round_id, entry in enumerate(manifest, start=1):
        if int(entry["round"]) != round_id or len(entry["clients"]) != int(config["num_clients"]):
            raise AssertionError("Invalid FedCAPS checkpoint manifest accounting")
        for relative in [entry["server"], *entry["clients"].values()]:
            path = output_dir / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"Missing checkpoint: {relative}")
    if len(rows) != int(config["rounds"]) * int(config["num_clients"]):
        raise AssertionError("FedCAPS must checkpoint and evaluate every client after every round")


def verify_evaluation_metrics(rows: list[dict], config: dict, global_test_rows: int) -> None:
    expected = int(config["rounds"]) * int(config["num_clients"])
    if len(rows) != expected:
        raise AssertionError(f"Evaluation rows {len(rows)} != {expected}")
    for row in rows:
        missing = [name for name in COMMON_METRIC_NAMES if name not in row]
        if missing or int(row["test_examples"]) != int(global_test_rows):
            raise AssertionError(f"Invalid evaluation row: missing={missing}")
        if row["model_scope"] != "client":
            raise AssertionError("FedCAPS evaluates personalized clients only on global_test")


def run_final_personalized_rounds(
    config: dict,
    paths: dict[str, Path],
    dataset_summary: dict,
    initial_states: dict,
    initialization_hashes: dict,
    model_metadata: dict,
    selected_feature_ids: list[int],
    selected_features: dict,
    encoder_payload: dict,
    ppo_result: dict,
    original_records: list[dict],
    task_queues,
    result_queue,
    workers,
    client_assignments: dict,
    logger: logging.Logger,
    utilization_samples: list[dict],
    cache_telemetry: dict,
    benchmark_results: dict,
    assignment_details: dict,
    pipeline_started: float,
    stage_times: dict,
) -> None:
    training_started = time.perf_counter()
    autoencoder_checkpoint = torch.load(encoder_payload["checkpoint_path"], map_location="cpu", weights_only=False)
    client_states = {
        str(client_id): clone_state(initial_states[config["client_models"][str(client_id)]])
        for client_id in range(1, int(config["num_clients"]) + 1)
    }
    local_history = []
    client_history = []
    round_history = []
    evaluation_rows = []
    checkpoint_manifest = []
    final_test_rows = []
    final_pooled_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    final_worker_seconds = {"0": 0.0, "1": 0.0}
    evaluation_worker_seconds = {"0": 0.0, "1": 0.0}
    total_test_seconds = 0.0
    training_assignment = client_assignments["final_training"]
    evaluation_assignment = client_assignments["evaluation"]
    logger.info("[STAGE final] Start %d personalized rounds; one full-data local epoch per round", int(config["rounds"]))
    for round_id in range(1, int(config["rounds"]) + 1):
        round_started = time.perf_counter()
        train_tasks = [
            (
                gpu_id,
                {
                    "task_id": f"round_{round_id:03d}_train_g{gpu_id}",
                    "kind": "train_final",
                    "round": round_id,
                    "client_ids": training_assignment[str(gpu_id)],
                    "feature_ids": selected_feature_ids,
                    "states": {str(client_id): client_states[str(client_id)] for client_id in training_assignment[str(gpu_id)]},
                },
            )
            for gpu_id in range(2)
        ]
        train_dispatch = dispatch(task_queues, result_queue, workers, train_tasks, int(config["worker_timeout_seconds"]))
        round_local_rows = []
        for result in train_dispatch:
            payload = result["payload"]
            worker_payload = torch.load(payload["payload_path"], map_location="cpu", weights_only=False)
            client_states.update(worker_payload["states"])
            round_local_rows.extend(worker_payload["history_local_epoch"])
            final_worker_seconds[str(result["gpu_id"])] += float(payload["seconds"])
            cache_telemetry["final_training"][str(result["gpu_id"])] = payload["cache"]
            utilization_samples.extend(result["utilization_samples"])
        round_local_rows.sort(key=lambda row: int(row["client_id"]))
        local_history.extend(round_local_rows)

        round_dir = paths["checkpoints"] / f"round_{round_id:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "round": round_id,
            "server_state_type": "fedcaps_feature_selection_stack_not_a_classifier",
            "encoder_state_dict": autoencoder_checkpoint["encoder_state_dict"],
            "decoder_state_dict": autoencoder_checkpoint["decoder_state_dict"],
            "actor_state_dict": ppo_result["actor_state_dict"],
            "critic_state_dict": ppo_result["critic_state_dict"],
            "reward_critic_state_dict": ppo_result["reward_critic_state_dict"],
            "selected_feature_ids": selected_feature_ids,
            "selected_feature_names": selected_features["selected_feature_names"],
            "server_classification_status": "not_applicable",
            "config": config,
            "feature_columns": FEATURE_COLUMNS,
            "label_mapping": dataset_summary["label_mapping"],
            "model_metadata": model_metadata,
            "initialization_hashes": initialization_hashes,
            "rng_metadata": {"seed": int(config["seed"]), "round": round_id},
        }, round_dir / "server.pt")
        client_paths = {}
        client_state_hashes = {}
        client_metric_lookups = {}
        for client_id in range(1, int(config["num_clients"]) + 1):
            relative = f"checkpoints/round_{round_id:03d}/client_{client_id:02d}.pt"
            client_hash = state_hash(client_states[str(client_id)])
            torch.save({
                "round": round_id,
                "client_id": client_id,
                "model_family": config["client_models"][str(client_id)],
                "model_state_dict": client_states[str(client_id)],
                "selected_feature_ids": selected_feature_ids,
                "train_rows": int(dataset_summary["clients"][str(client_id)]["train_rows"]),
                "local_validation_rows": 0,
                "state_hash": client_hash,
                "config": config,
                "feature_columns": FEATURE_COLUMNS,
                "label_mapping": dataset_summary["label_mapping"],
                "model_metadata": model_metadata[config["client_models"][str(client_id)]],
                "rng_metadata": {"seed": int(config["seed"]), "round": round_id, "client_id": client_id},
            }, paths["root"] / relative)
            client_paths[str(client_id)] = relative
            client_state_hashes[str(client_id)] = client_hash
            client_metric_lookups[str(client_id)] = (
                f"metrics/evaluation_metrics.csv#round={round_id}&model_scope=client&client_id={client_id}"
            )
        checkpoint_manifest.append({
            "round": round_id,
            "server": f"checkpoints/round_{round_id:03d}/server.pt",
            "server_state_type": "fedcaps_feature_selection_stack_not_a_classifier",
            "server_classification_status": "not_applicable",
            "server_state_hashes": {
                "encoder": state_hash(autoencoder_checkpoint["encoder_state_dict"]),
                "decoder": state_hash(autoencoder_checkpoint["decoder_state_dict"]),
                "actor": state_hash(ppo_result["actor_state_dict"]),
                "critic": state_hash(ppo_result["critic_state_dict"]),
                "reward_critic": state_hash(ppo_result["reward_critic_state_dict"]),
            },
            "clients": client_paths,
            "client_state_hashes": client_state_hashes,
            "client_metric_lookups": client_metric_lookups,
            "optimizer_state_persistence": "optimizer_recreated_for_each_local_epoch",
        })
        atomic_json(paths["checkpoints"] / "checkpoint_manifest.json", checkpoint_manifest)

        test_started = time.perf_counter()
        test_tasks = [
            (
                gpu_id,
                {
                    "task_id": f"round_{round_id:03d}_test_g{gpu_id}",
                    "kind": "test_final",
                    "round": round_id,
                    "client_ids": evaluation_assignment[str(gpu_id)],
                    "feature_ids": selected_feature_ids,
                    "states": {str(client_id): client_states[str(client_id)] for client_id in evaluation_assignment[str(gpu_id)]},
                },
            )
            for gpu_id in range(2)
        ]
        test_dispatch = dispatch(task_queues, result_queue, workers, test_tasks, int(config["worker_timeout_seconds"]))
        round_test_rows = []
        for result in test_dispatch:
            round_test_rows.extend(json.loads(Path(result["payload"]["payload_path"]).read_text(encoding="utf-8")))
            evaluation_worker_seconds[str(result["gpu_id"])] += float(result["payload"]["seconds"])
            cache_telemetry["final_training"][str(result["gpu_id"])] = result["payload"]["cache"]
            utilization_samples.extend(result["utilization_samples"])
        round_test_rows.sort(key=lambda row: int(row["client_id"]))
        if len(round_test_rows) != int(config["num_clients"]):
            raise AssertionError("Every personalized client must be evaluated after every round")
        total_test_seconds += time.perf_counter() - test_started
        raw_dir = paths["metrics"] / "confusion_matrices" / f"round_{round_id:03d}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        lookup = {}
        for result in round_test_rows:
            client_id = int(result["client_id"])
            confusion = np.asarray(result["confusion_matrix"], dtype=np.int64)
            np.save(raw_dir / f"client_{client_id:02d}.npy", confusion)
            pd.DataFrame(confusion).to_csv(raw_dir / f"client_{client_id:02d}.csv", index=False)
            metric_row = {
                "method": config["method"],
                "scenario": config["scenario"],
                "run_name": config["run_name"],
                "round": round_id,
                "model_scope": "client",
                "client_id": client_id,
                "checkpoint_relative_path": client_paths[str(client_id)],
                "test_examples": int(result["metrics"]["examples"]),
                **{name: float(result["metrics"][name]) for name in COMMON_METRIC_NAMES},
                "confusion_matrix_relative_path": str((raw_dir / f"client_{client_id:02d}.npy").relative_to(paths["root"])),
            }
            evaluation_rows.append(metric_row)
            lookup[client_id] = metric_row
        for train_row in round_local_rows:
            metric_row = lookup[int(train_row["client_id"])]
            client_history.append({**train_row, **{f"test_{name}": metric_row[name] for name in COMMON_METRIC_NAMES}})
        mean_train_loss = float(np.mean([row["total_loss"] for row in round_local_rows]))
        mean_metrics = {
            name: float(np.mean([lookup[client_id][name] for client_id in lookup]))
            for name in COMMON_METRIC_NAMES
        }
        round_history.append({
            "round": round_id,
            "clients_trained": int(config["num_clients"]),
            "local_epochs_per_client": 1,
            "train_rows": int(sum(row["train_examples"] for row in round_local_rows)),
            "mean_train_loss": mean_train_loss,
            **{f"mean_client_{name}": value for name, value in mean_metrics.items()},
            "round_seconds": time.perf_counter() - round_started,
        })
        if round_id == int(config["rounds"]):
            final_test_rows = round_test_rows
            final_pooled_confusion = sum(
                (np.asarray(row["confusion_matrix"], dtype=np.int64) for row in round_test_rows),
                start=np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
            )
        logger.info(
            "[STAGE final] round=%d/%d mean_train_loss=%.6f mean_accuracy=%.6f mean_macro_f1=%.6f seconds=%.2f",
            round_id, int(config["rounds"]), mean_train_loss, mean_metrics["accuracy"],
            mean_metrics["macro_f1"], round_history[-1]["round_seconds"],
        )

    final_training_seconds = time.perf_counter() - training_started
    assignment_details.setdefault("actual_worker_seconds_by_phase", {})["final_training"] = final_worker_seconds
    assignment_details.setdefault("actual_idle_seconds_by_phase", {})["final_training"] = {
        gpu_id: max(final_worker_seconds.values()) - seconds
        for gpu_id, seconds in final_worker_seconds.items()
    }
    assignment_details.setdefault("actual_worker_seconds_by_phase", {})["evaluation"] = evaluation_worker_seconds
    assignment_details.setdefault("actual_idle_seconds_by_phase", {})["evaluation"] = {
        gpu_id: max(evaluation_worker_seconds.values()) - seconds
        for gpu_id, seconds in evaluation_worker_seconds.items()
    }
    local_history.sort(key=lambda row: (int(row["round"]), int(row["client_id"])))
    client_history.sort(key=lambda row: (int(row["round"]), int(row["client_id"])))
    rows_to_csv_json(round_history, paths["metrics"] / "history_round.csv", paths["metrics"] / "history_round.json")
    rows_to_csv_json(client_history, paths["metrics"] / "history_client.csv", paths["metrics"] / "history_client.json")
    rows_to_csv_json(local_history, paths["metrics"] / "history_local_epoch.csv", paths["metrics"] / "history_local_epoch.json")
    rows_to_csv_json(evaluation_rows, paths["metrics"] / "evaluation_metrics.csv", paths["metrics"] / "evaluation_metrics.json")
    final_round_metrics = [row for row in evaluation_rows if int(row["round"]) == int(config["rounds"])]
    rows_to_csv_json(final_round_metrics, paths["metrics"] / "final_round_metrics.csv", paths["metrics"] / "final_round_metrics.json")
    personalized_csv = [{"client_id": row["client_id"], "model_family": row["model_family"], **row["metrics"]} for row in final_test_rows]
    rows_to_csv_json(personalized_csv, paths["metrics"] / "personalized_test_metrics.csv", paths["metrics"] / "personalized_test_metrics.json")

    expected_support = int(config["num_clients"]) * int(dataset_summary["global_test"]["rows"])
    if int(final_pooled_confusion.sum()) != expected_support:
        raise AssertionError("Final pooled confusion support mismatch")
    np.save(paths["metrics"] / "confusion_matrix.npy", final_pooled_confusion)
    pd.DataFrame(final_pooled_confusion).to_csv(paths["metrics"] / "confusion_matrix.csv", index=False)
    report_rows, report_aggregate = classification_report_from_confusion(final_pooled_confusion, dataset_summary["label_mapping"])
    pd.DataFrame(report_rows).to_csv(paths["metrics"] / "classification_report.csv", index=False)
    atomic_json(paths["metrics"] / "classification_report.json", {"classes": report_rows, **report_aggregate})
    weighted_test_metrics = weighted_metric_average(final_test_rows, dataset_summary, True)
    unweighted_test_metrics = weighted_metric_average(final_test_rows, dataset_summary, False)

    record_upload_bytes = sum(2 * len(record["feature_ids"]) + 12 for record in original_records)
    communication_total = record_upload_bytes + int(ppo_result["communication_bytes"])
    communication_rows = []
    for round_id in range(1, int(config["rounds"]) + 1):
        communication_rows.append({
            "round": round_id,
            "bytes_this_round": communication_total if round_id == 1 else 0,
            "bytes_cumulative": communication_total,
            "mib_cumulative": communication_total / (1024**2),
            "classifier_parameter_communication_bytes": 0,
        })
    communication = {
        "feature_record_upload_bytes": record_upload_bytes,
        "ppo_candidate_feedback_bytes": int(ppo_result["communication_bytes"]),
        "classifier_parameter_communication_bytes": 0,
        "total_bytes": communication_total,
        "server_classifier_status": "not_applicable",
    }
    atomic_json(paths["metrics"] / "communication_costs.json", communication)
    pd.DataFrame(communication_rows).to_csv(paths["metrics"] / "communication_costs.csv", index=False)
    rows_to_csv_json(communication_rows, paths["metrics"] / "communication_cumulative.csv", paths["metrics"] / "communication_cumulative.json")
    rows_to_csv_json(round_history, paths["metrics"] / "runtime_per_round.csv", paths["metrics"] / "runtime_per_round.json")
    rows_to_csv_json(utilization_samples, paths["metrics"] / "gpu_utilization.csv", paths["metrics"] / "gpu_utilization.json")

    plotting_seconds = save_plots(
        paths, dataset_summary, local_history, ppo_result["history"], ppo_result["candidate_evaluations"],
        final_pooled_confusion, report_rows, selected_feature_ids, evaluation_rows, round_history, utilization_samples,
    )
    runtime = {
        **stage_times,
        "final_training_and_per_round_test_seconds": final_training_seconds,
        "final_training_worker_seconds": final_worker_seconds,
        "global_test_evaluation_seconds": total_test_seconds,
        "plotting_seconds": plotting_seconds,
        "entry_point_seconds": time.perf_counter() - pipeline_started,
    }
    atomic_json(paths["metrics"] / "runtime_breakdown.json", runtime)
    environment = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [
            {
                "gpu_id": gpu_id,
                "name": torch.cuda.get_device_name(gpu_id),
                "total_vram_bytes": int(torch.cuda.get_device_properties(gpu_id).total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(gpu_id)),
            }
            for gpu_id in range(2)
        ],
    }
    summary = {
        "run_name": config["run_name"],
        "method": config["method"],
        "scenario": config["scenario_name"],
        "status": "complete",
        "rounds": int(config["rounds"]),
        "environment": environment,
        "model_metadata": model_metadata,
        "initialization_hashes": initialization_hashes,
        "dataset": {
            "global_train_rows": dataset_summary["global_train_rows"],
            "global_test_rows": dataset_summary["global_test"]["rows"],
            "split_policy": dataset_summary["split_policy"],
            "feature_search_cv_folds": int(config["fedcaps_search_cv_folds"]),
        },
        "assignment": assignment_details,
        "benchmark": benchmark_results,
        "cache": cache_telemetry,
        "server_evaluation": {"status": "not_applicable", "reason": "FedCAPS server state is not a 34-logit classifier"},
        "selected_features": selected_features,
        "final_weighted_mean_personalized_test_metrics": weighted_test_metrics,
        "final_unweighted_mean_personalized_test_metrics": unweighted_test_metrics,
        "final_pooled_personalized_test_metrics": metrics_from_confusion(final_pooled_confusion),
        "runtime": runtime,
        "communication": communication,
        "outputs": {
            "checkpoint_manifest": "checkpoints/checkpoint_manifest.json",
            "evaluation_metrics": "metrics/evaluation_metrics.csv",
            "final_round_metrics": "metrics/final_round_metrics.csv",
            "selected_features": "metrics/selected_features.json",
        },
    }
    atomic_json(paths["metrics"] / "server_evaluation_status.json", {
        "status": "not_applicable",
        "reason": "FedCAPS server state is a feature-selection stack and does not emit 34-class logits",
        "evaluated_clients_each_round": int(config["num_clients"]),
    })
    atomic_json(paths["metrics"] / "summary.json", summary)
    verify_checkpoint_manifest(paths["root"], config, evaluation_rows)
    verify_evaluation_metrics(evaluation_rows, config, int(dataset_summary["global_test"]["rows"]))
    missing_or_empty = [str(path) for path in required_output_paths(paths) if not path.is_file() or path.stat().st_size == 0]
    if missing_or_empty:
        raise AssertionError(f"Missing or empty required outputs: {missing_or_empty}")
    logger.info(
        "Completed %s selected_features=%d evaluation_rows=%d seconds=%.2f output_dir=%s",
        config["run_name"], len(selected_feature_ids), len(evaluation_rows),
        time.perf_counter() - pipeline_started, paths["root"],
    )


def main() -> None:
    pipeline_started = time.perf_counter()
    manifest = read_manifest()
    config = manifest["config"]
    paths = ensure_output_dirs(config)
    logger = setup_logger(paths["logs"] / "run.log", "fedcaps-coordinator")
    logger.info("Starting %s scenario=%s", config["run_name"], config["scenario_name"])
    logger.info(
        "Configuration clients=%d batch=%d MARLFS_epochs=%d permutations=%d PPO_epochs=%d PPO_steps=%d rounds=%d",
        int(config["num_clients"]),
        int(config["per_client_batch_size"]),
        int(config["marlfs_epochs"]),
        int(config["record_permutations"]),
        int(config["ppo_search_epochs"]),
        int(config["ppo_steps_per_epoch"]),
        int(config["rounds"]),
    )
    set_all_seeds(int(config["seed"]))
    atomic_json(paths["metrics"] / "config.json", config)
    rows_to_csv_json(
        [{
            "enabled": False,
            "reason": "FedCAPS has no server classifier pretraining stage",
            "train_examples": 0,
            "optimizer_steps": 0,
            "seconds": 0.0,
        }],
        paths["metrics"] / "history_pretrain.csv",
        paths["metrics"] / "history_pretrain.json",
    )
    data_started = time.perf_counter()
    dataset_summary = prepare_data(config, paths, logger)
    data_seconds = time.perf_counter() - data_started
    initial_states = {}
    initialization_hashes = {}
    model_metadata = {}
    for family in sorted(set(config["client_models"].values())):
        set_all_seeds(int(config["initialization_seed"]))
        model = build_classifier(family)
        initial_states[family] = cpu_state(model)
        initialization_hashes[family] = state_hash(initial_states[family])
        model_metadata[family] = {
            "parameter_count": count_parameters(model),
            "state_bytes": sum(value.numel() * value.element_size() for value in initial_states[family].values()),
            "output_shape": [None, NUM_CLASSES],
        }
    torch.save(initial_states, paths["temporary"] / "initial_states.pt")
    manifest["dataset_summary"] = dataset_summary
    manifest["paths"] = {name: str(path) for name, path in paths.items()}
    atomic_json(Path(os.environ["TRAINING_MANIFEST"]), manifest)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(2)]
    manifest_path = os.environ["TRAINING_MANIFEST"]
    workers = [
        context.Process(target=worker_main, args=(gpu_id, task_queues[gpu_id], result_queue, manifest_path))
        for gpu_id in range(2)
    ]
    for worker in workers:
        worker.start()
    utilization_samples = []
    try:
        active_families = sorted(set(config["client_models"].values()))
        benchmark_started = time.perf_counter()
        logger.info("[STAGE benchmark] Start active_families=%s stream_candidates=%s", active_families, config["stream_candidates"])
        benchmark_tasks = [
            (gpu_id, {"task_id": f"benchmark_g{gpu_id}", "kind": "benchmark", "families": active_families})
            for gpu_id in range(2)
        ]
        benchmark_dispatch = dispatch(task_queues, result_queue, workers, benchmark_tasks, int(config["worker_timeout_seconds"]))
        benchmark_results = {}
        for result in benchmark_dispatch:
            benchmark_results[str(result["gpu_id"])] = result["payload"]
            utilization_samples.extend(result["utilization_samples"])
        client_assignments, assignment_details = select_client_assignments(dataset_summary, benchmark_results, config)
        search_assignment = client_assignments["search"]
        logger.info(
            "[STAGE benchmark] Complete search_assignment_gpu0=%s search_assignment_gpu1=%s predicted_by_phase=%s stream_runtime=%s stream_benchmark_recommendation=%s",
            search_assignment["0"],
            search_assignment["1"],
            assignment_details["predicted_by_phase"],
            assignment_details["stream_selection_by_gpu_and_family"],
            assignment_details["stream_benchmark_recommendation_by_gpu_and_family"],
        )
        atomic_json(paths["metrics"] / "config.json", config)
        cache_tasks = [
            (
                gpu_id,
                {
                    "task_id": f"cache_g{gpu_id}",
                    "kind": "initialize_cache",
                    "phase": "search",
                    "client_ids": search_assignment[str(gpu_id)],
                    "runtime_stream_count_by_gpu": config["runtime_stream_count_by_gpu"],
                },
            )
            for gpu_id in range(2)
        ]
        cache_dispatch = dispatch(task_queues, result_queue, workers, cache_tasks, int(config["worker_timeout_seconds"]))
        cache_telemetry = {"search": {}, "final_training": {}}
        for result in cache_dispatch:
            cache_telemetry["search"][str(result["gpu_id"])] = result["payload"]
            utilization_samples.extend(result["utilization_samples"])
        benchmark_cache_seconds = time.perf_counter() - benchmark_started
        logger.info("[STAGE cache] Complete telemetry=%s seconds=%.2f", cache_telemetry, benchmark_cache_seconds)
        marlfs_started = time.perf_counter()
        logger.info("[STAGE MARLFS] Dispatching collectors to both GPUs")
        collect_tasks = [
            (gpu_id, {"task_id": f"collect_g{gpu_id}", "kind": "collect_records", "client_ids": search_assignment[str(gpu_id)]})
            for gpu_id in range(2)
        ]
        collect_dispatch = dispatch(task_queues, result_queue, workers, collect_tasks, int(config["worker_timeout_seconds"]))
        original_records = []
        marlfs_worker_seconds = {}
        for result in collect_dispatch:
            payload = result["payload"]
            original_records.extend(json.loads(Path(payload["records_path"]).read_text(encoding="utf-8")))
            marlfs_worker_seconds[str(result["gpu_id"])] = float(payload["seconds"])
            utilization_samples.extend(result["utilization_samples"])
        expected_original_records = 10 * int(config["marlfs_epochs"])
        if len(original_records) != expected_original_records:
            raise AssertionError(f"MARLFS records {len(original_records)} != {expected_original_records}")
        original_records = sorted(original_records, key=lambda row: (int(row["client_id"]), int(row["collection_epoch"])))
        logger.info(
            "[STAGE MARLFS] Collection complete records=%s worker_seconds=%s",
            f"{len(original_records):,}",
            marlfs_worker_seconds,
        )
        assignment_details.setdefault("actual_worker_seconds_by_phase", {})["marlfs"] = marlfs_worker_seconds
        assignment_details.setdefault("actual_idle_seconds_by_phase", {})["marlfs"] = {
            gpu_id: max(marlfs_worker_seconds.values()) - seconds
            for gpu_id, seconds in marlfs_worker_seconds.items()
        }
        augmentation_started = time.perf_counter()
        logger.info("[STAGE augmentation] Start permutations_per_record=%d", int(config["record_permutations"]))
        augmented_records = augment_records(original_records, int(config["record_permutations"]), int(config["seed"]))
        expected_augmented = expected_original_records * int(config["record_permutations"])
        if len(augmented_records) != expected_augmented:
            raise AssertionError(f"Augmented records {len(augmented_records)} != {expected_augmented}")
        pd.DataFrame(records_for_csv(augmented_records)).to_csv(paths["metrics"] / "feature_selection_records.csv", index=False)
        atomic_json(paths["metrics"] / "feature_selection_records.json", augmented_records)
        augmented_path = paths["temporary"] / "augmented_records.json"
        atomic_json(augmented_path, augmented_records)
        augmentation_seconds = time.perf_counter() - augmentation_started
        logger.info(
            "[STAGE augmentation] Complete augmented_records=%s seconds=%.2f",
            f"{len(augmented_records):,}",
            augmentation_seconds,
        )
        marlfs_seconds = time.perf_counter() - marlfs_started
        logger.info("[STAGE encoder] Dispatching server encoder-decoder training to GPU 0")
        encoder_tasks = [(0, {"task_id": "train_autoencoder_g0", "kind": "train_autoencoder", "records_path": str(augmented_path)})]
        encoder_dispatch = dispatch(task_queues, result_queue, workers, encoder_tasks, int(config["worker_timeout_seconds"]))
        encoder_payload = encoder_dispatch[0]["payload"]
        utilization_samples.extend(encoder_dispatch[0]["utilization_samples"])
        rows_to_csv_json(
            encoder_payload["history"],
            paths["metrics"] / "encoder_history.csv",
            paths["metrics"] / "encoder_history.json",
        )
        logger.info(
            "[STAGE encoder] Result best_epoch=%d best_validation_loss=%.6f invariance_error=%.9f seconds=%.2f",
            int(encoder_payload["best_epoch"]),
            float(encoder_payload["best_validation_loss"]),
            float(encoder_payload["permutation_invariance_max_abs_error"]),
            float(encoder_payload["seconds"]),
        )
        ppo_result = run_ppo_search(
            config,
            dataset_summary,
            original_records,
            Path(encoder_payload["checkpoint_path"]),
            task_queues,
            result_queue,
            workers,
            search_assignment,
            logger,
        )
        rows_to_csv_json(ppo_result["history"], paths["metrics"] / "ppo_history.csv", paths["metrics"] / "ppo_history.json")
        candidate_csv = []
        for row in ppo_result["candidate_evaluations"]:
            candidate_csv.append({
                **row,
                "feature_ids": json.dumps(row["feature_ids"], default=json_compatible),
                "feature_names": json.dumps(row["feature_names"], ensure_ascii=False, default=json_compatible),
                "per_client": json.dumps(row["per_client"], default=json_compatible),
            })
        pd.DataFrame(candidate_csv).to_csv(paths["metrics"] / "candidate_evaluations.csv", index=False)
        selected_feature_ids = [int(value) for value in ppo_result["selected_feature_ids"]]
        selected_features = {
            "selected_feature_ids": selected_feature_ids,
            "selected_feature_names": [FEATURE_COLUMNS[value] for value in selected_feature_ids],
            "selected_feature_count": len(selected_feature_ids),
            "best_by_reward": ppo_result["best_by_reward"],
            "best_by_weighted_micro_f1": ppo_result["best_by_weighted_micro_f1"],
        }
        atomic_json(paths["metrics"] / "selected_features.json", selected_features)
        logger.info(
            "[STAGE selection] Selected feature_count=%d ids=%s names=%s reward=%.6f weighted_micro_f1=%.6f",
            len(selected_feature_ids),
            selected_feature_ids,
            selected_features["selected_feature_names"],
            float(ppo_result["best_by_reward"]["reward"]),
            float(ppo_result["best_by_reward"]["weighted_micro_f1"]),
        )
        ppo_worker_seconds = {"0": 0.0, "1": 0.0}
        for candidate in ppo_result["candidate_evaluations"]:
            for evaluation in candidate["per_client"]:
                ppo_worker_seconds[str(evaluation["gpu_id"])] += float(evaluation["seconds"])
        assignment_details.setdefault("actual_worker_seconds_by_phase", {})["ppo_feedback"] = ppo_worker_seconds
        assignment_details.setdefault("actual_idle_seconds_by_phase", {})["ppo_feedback"] = {
            gpu_id: max(ppo_worker_seconds.values()) - seconds
            for gpu_id, seconds in ppo_worker_seconds.items()
        }

        final_cache_started = time.perf_counter()
        final_training_assignment = client_assignments["final_training"]
        logger.info(
            "[STAGE cache] Rebalancing ownership for final training GPU0=%s GPU1=%s",
            final_training_assignment["0"],
            final_training_assignment["1"],
        )
        final_cache_tasks = [
            (
                gpu_id,
                {
                    "task_id": f"final_cache_g{gpu_id}",
                    "kind": "initialize_cache",
                    "phase": "final_training",
                    "client_ids": final_training_assignment[str(gpu_id)],
                    "runtime_stream_count_by_gpu": config["runtime_stream_count_by_gpu"],
                },
            )
            for gpu_id in range(2)
        ]
        final_cache_dispatch = dispatch(
            task_queues,
            result_queue,
            workers,
            final_cache_tasks,
            int(config["worker_timeout_seconds"]),
        )
        for result in final_cache_dispatch:
            cache_telemetry["final_training"][str(result["gpu_id"])] = result["payload"]
            utilization_samples.extend(result["utilization_samples"])
        final_cache_rebalance_seconds = time.perf_counter() - final_cache_started
        run_final_personalized_rounds(
            config=config,
            paths=paths,
            dataset_summary=dataset_summary,
            initial_states=initial_states,
            initialization_hashes=initialization_hashes,
            model_metadata=model_metadata,
            selected_feature_ids=selected_feature_ids,
            selected_features=selected_features,
            encoder_payload=encoder_payload,
            ppo_result=ppo_result,
            original_records=original_records,
            task_queues=task_queues,
            result_queue=result_queue,
            workers=workers,
            client_assignments=client_assignments,
            logger=logger,
            utilization_samples=utilization_samples,
            cache_telemetry=cache_telemetry,
            benchmark_results=benchmark_results,
            assignment_details=assignment_details,
            pipeline_started=pipeline_started,
            stage_times={
                "preprocessing_seconds": data_seconds,
                "benchmark_and_cache_seconds": benchmark_cache_seconds,
                "marlfs_seconds": marlfs_seconds,
                "marlfs_worker_seconds": marlfs_worker_seconds,
                "permutation_augmentation_seconds": augmentation_seconds,
                "encoder_decoder_seconds": float(encoder_payload["seconds"]),
                "ppo_search_and_feedback_seconds": float(ppo_result["seconds"]),
                "final_cache_rebalance_seconds": final_cache_rebalance_seconds,
            },
        )
        return
    finally:
        for gpu_id in range(2):
            if workers[gpu_id].is_alive():
                task_queues[gpu_id].put({"task_id": f"shutdown_g{gpu_id}", "kind": "shutdown"})
        for worker in workers:
            worker.join(timeout=30)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        bad_exit_codes = [worker.exitcode for worker in workers if worker.exitcode != 0]
        if bad_exit_codes:
            raise RuntimeError(f"Worker nonzero exit codes: {bad_exit_codes}")


if __name__ == "__main__":
    main()
