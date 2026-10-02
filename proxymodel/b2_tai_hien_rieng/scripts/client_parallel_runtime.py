from __future__ import annotations

import hashlib
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


MANIFEST_PATH = Path(os.environ["TRAINING_MANIFEST"])
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
CONFIG = MANIFEST["config"]
DATA = MANIFEST["confirmed_data"]
OUT = Path(MANIFEST["out_dir"])
CACHE_DIR = Path(MANIFEST["cache_dir"])
FEATURE_COLUMNS = CONFIG["feature_columns"]
LABEL_MAPPING = {int(key): value for key, value in DATA["label_mapping"].items()}
BENIGN_CLASS_ID = int(DATA["benign_class_id"])
MIB = 1024 ** 2


class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.classifier = nn.Linear(64, CONFIG["num_classes"])

    def forward(self, features):
        sequence = features.unsqueeze(-1)
        encoded, _ = self.gru(sequence)
        return self.classifier(encoded[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_projection = nn.Linear(1, 64)
        self.position = nn.Parameter(torch.empty(1, CONFIG["num_features"], 64))
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
        self.classifier = nn.Linear(64, CONFIG["num_classes"])
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, features):
        tokens = self.scalar_projection(features.unsqueeze(-1)) + self.position
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded.mean(dim=1)))


class CNN1DProxy(nn.Module):
    def __init__(self):
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
        self.classifier = nn.Linear(128, CONFIG["num_classes"])

    def forward(self, features):
        encoded = self.features(features.unsqueeze(1)).squeeze(-1)
        return self.classifier(encoded)


def build_model(family):
    if family == "gru":
        return GRUClassifier()
    if family == "transformer":
        return TransformerClassifier()
    if family == "cnn1d":
        return CNN1DProxy()
    raise ValueError(f"Unknown model family: {family}")


def cpu_state(model):
    return OrderedDict(
        (key, tensor.detach().cpu().clone())
        for key, tensor in model.state_dict().items()
    )


def clone_state(state):
    return OrderedDict((key, tensor.clone()) for key, tensor in state.items())


def make_initial_state(family):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CONFIG["initialization_seed"])
        model = build_model(family)
    return cpu_state(model)


def state_sha256(state):
    digest = hashlib.sha256()
    for key, tensor in state.items():
        digest.update(key.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_metadata(family, state):
    model = build_model(family)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    return {
        "model_family": family,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_bytes": trainable_bytes,
        "model_state_bytes": state_bytes,
        "model_state_mib": state_bytes / MIB,
        "initialization_sha256": state_sha256(state),
    }


def safe_divide(numerator, denominator):
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
    total = float(matrix.sum())
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    true_negative = total - true_positive - false_positive - false_negative
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    fpr = safe_divide(false_positive, false_positive + true_negative)
    fnr = safe_divide(false_negative, false_negative + true_positive)
    weights = safe_divide(support, np.full_like(support, max(total, 1.0)))

    benign_tp = float(matrix[BENIGN_CLASS_ID, BENIGN_CLASS_ID])
    benign_as_attack = float(matrix[BENIGN_CLASS_ID].sum() - benign_tp)
    attack_as_benign = float(matrix[:, BENIGN_CLASS_ID].sum() - benign_tp)
    attack_tp = total - benign_tp - benign_as_attack - attack_as_benign
    attack_precision = attack_tp / max(attack_tp + benign_as_attack, 1.0)
    attack_recall = attack_tp / max(attack_tp + attack_as_benign, 1.0)
    attack_f1 = (
        2.0 * attack_precision * attack_recall
        / max(attack_precision + attack_recall, 1e-15)
    )
    binary_accuracy = (attack_tp + benign_tp) / max(total, 1.0)
    binary_fpr = benign_as_attack / max(benign_as_attack + benign_tp, 1.0)
    binary_fnr = attack_as_benign / max(attack_as_benign + attack_tp, 1.0)
    return {
        "accuracy": float(true_positive.sum() / max(total, 1.0)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "multiclass_macro_fpr": float(fpr.mean()),
        "multiclass_macro_fnr": float(fnr.mean()),
        "binary_attack_accuracy": float(binary_accuracy),
        "binary_attack_precision": float(attack_precision),
        "binary_attack_recall": float(attack_recall),
        "binary_attack_f1": float(attack_f1),
        "binary_attack_fpr": float(binary_fpr),
        "binary_attack_fnr": float(binary_fnr),
    }


def prefixed_metrics(prefix, evaluation):
    values = {f"{prefix}_loss": evaluation["loss"]}
    values.update(
        {f"{prefix}_{name}": value for name, value in evaluation["metrics"].items()}
    )
    return values


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=False),
        encoding="utf-8",
    )


def classification_report_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = safe_divide(true_positive, predicted)
    recall = safe_divide(true_positive, support)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    rows = []
    report = {}
    for class_id in range(CONFIG["num_classes"]):
        row = {
            "class_id": class_id,
            "class_name": LABEL_MAPPING[class_id],
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        rows.append(row)
        report[str(class_id)] = row
    total = float(support.sum())
    weights = safe_divide(support, np.full_like(support, max(total, 1.0)))
    for name, values in (
        ("macro avg", np.ones_like(support) / len(support)),
        ("weighted avg", weights),
    ):
        row = {
            "class_id": name,
            "class_name": name,
            "precision": float((precision * values).sum()),
            "recall": float((recall * values).sum()),
            "f1": float((f1 * values).sum()),
            "support": int(total),
        }
        rows.append(row)
        report[name] = row
    return report, rows


def save_history(name, rows):
    pd.DataFrame(rows).to_csv(OUT / "metrics" / f"{name}.csv", index=False)
    write_json(OUT / "metrics" / f"{name}.json", rows)


def aggregate_proxy_states(states, weights):
    total_weight = float(sum(weights))
    aggregated = OrderedDict()
    for key in states[0]:
        tensors = [state[key] for state in states]
        if tensors[0].is_floating_point():
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float64)
            for tensor, weight in zip(tensors, weights):
                accumulator.add_(tensor.to(torch.float64), alpha=weight / total_weight)
            aggregated[key] = accumulator.to(tensors[0].dtype)
        else:
            aggregated[key] = torch.stack(tensors).max(dim=0).values
    return aggregated


def derive_seed(round_index, client_id, phase):
    phase_code = {
        "mutual": 11,
        "finetune": 29,
        "validation": 43,
        "test": 61,
    }[phase]
    return (
        CONFIG["seed"] * 1_000_003
        + round_index * 10_007
        + client_id * 101
        + phase_code
    ) % (2 ** 31 - 1)


def setup_logger(log_path, name, include_stdout=False):
    handlers = [logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    if include_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)


def numpy_file_bytes(path):
    array = np.load(path, mmap_mode="r")
    return int(array.nbytes)


def torch_dtype_for_numpy(dtype):
    mapping = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int64"): torch.int64,
        np.dtype("int32"): torch.int32,
        np.dtype("uint8"): torch.uint8,
    }
    if np.dtype(dtype) not in mapping:
        raise TypeError(f"Unsupported cache dtype: {dtype}")
    return mapping[np.dtype(dtype)]


def validate_numpy_classification_cache(
    features_path,
    labels_path,
    owner,
    index_paths=None,
):
    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    expected_feature_shape = (len(labels), CONFIG["num_features"])
    if features.shape != expected_feature_shape:
        raise ValueError(
            f"{owner} feature shape must be {expected_feature_shape}, "
            f"observed {features.shape}"
        )
    if features.dtype != np.dtype("float32"):
        raise TypeError(
            f"{owner} features must use float32, observed {features.dtype}"
        )
    if labels.ndim != 1 or labels.dtype != np.dtype("int64"):
        raise TypeError(
            f"{owner} labels must be one-dimensional int64, "
            f"observed shape={labels.shape} dtype={labels.dtype}"
        )
    if labels.size:
        label_min = int(labels.min())
        label_max = int(labels.max())
        if label_min < 0 or label_max >= CONFIG["num_classes"]:
            raise ValueError(
                f"{owner} labels must be in [0, {CONFIG['num_classes'] - 1}], "
                f"observed [{label_min}, {label_max}]"
            )

    indexed_rows = 0
    for index_name, index_path in (index_paths or {}).items():
        indices = np.load(index_path, mmap_mode="r")
        if indices.ndim != 1 or indices.dtype != np.dtype("int64"):
            raise TypeError(
                f"{owner} {index_name} must be one-dimensional int64, "
                f"observed shape={indices.shape} dtype={indices.dtype}"
            )
        if indices.size:
            index_min = int(indices.min())
            index_max = int(indices.max())
            if index_min < 0 or index_max >= len(labels):
                raise IndexError(
                    f"{owner} {index_name} must index [0, {len(labels) - 1}], "
                    f"observed [{index_min}, {index_max}]"
                )
        indexed_rows += len(indices)
    if index_paths and indexed_rows != len(labels):
        raise ValueError(
            f"{owner} train/validation indices cover {indexed_rows} rows, "
            f"expected {len(labels)}"
        )


def load_numpy_to_device(path, device, chunk_rows, copy_stream):
    source = np.load(path, mmap_mode="r")
    destination = torch.empty(
        source.shape,
        dtype=torch_dtype_for_numpy(source.dtype),
        device=device,
    )
    total_rows = source.shape[0] if source.ndim else 1
    for start in range(0, total_rows, chunk_rows):
        end = min(total_rows, start + chunk_rows)
        cpu_array = np.array(source[start:end], copy=True)
        cpu_tensor = torch.from_numpy(cpu_array)
        pinned = torch.empty_like(cpu_tensor, pin_memory=True)
        pinned.copy_(cpu_tensor)
        with torch.cuda.stream(copy_stream):
            destination[start:end].copy_(pinned, non_blocking=True)
        copy_stream.synchronize()
    return destination


class GpuClientCache:
    def __init__(self, gpu_id, assigned_clients, device, logger):
        self.gpu_id = gpu_id
        self.assigned_clients = tuple(assigned_clients)
        self.device = device
        self.logger = logger
        self.copy_stream = torch.cuda.Stream(device=device)
        self.total_vram_bytes = torch.cuda.get_device_properties(gpu_id).total_memory
        self.cache_budget_bytes = int(
            self.total_vram_bytes * CONFIG["gpu_cache_fraction"]
        )
        self.client_bytes = {
            client_id: self._estimate_client_bytes(client_id)
            for client_id in self.assigned_clients
        }
        self.planned_bytes = sum(self.client_bytes.values())
        self.entries = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.validated_clients = set()
        self.mode = (
            "full_gpu_cache"
            if self.planned_bytes <= self.cache_budget_bytes
            else "deterministic_lru_gpu_cache"
        )
        if self.mode == "full_gpu_cache":
            self.ensure(self.assigned_clients)

    def _paths(self, client_id):
        item = DATA["clients"][str(client_id)]
        return {
            "features": item["features_path"],
            "labels": item["labels_path"],
            "train_indices": item["train_indices_path"],
            "val_indices": item["val_indices_path"],
        }

    def _estimate_client_bytes(self, client_id):
        return sum(numpy_file_bytes(path) for path in self._paths(client_id).values())

    def _load_client(self, client_id):
        paths = self._paths(client_id)
        if client_id not in self.validated_clients:
            validate_numpy_classification_cache(
                paths["features"],
                paths["labels"],
                f"client {client_id}",
                {
                    "train_indices": paths["train_indices"],
                    "val_indices": paths["val_indices"],
                },
            )
            self.validated_clients.add(client_id)
        loaded = {
            name: load_numpy_to_device(
                path,
                self.device,
                CONFIG["gpu_cache_chunk_rows"],
                self.copy_stream,
            )
            for name, path in paths.items()
        }
        actual_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in loaded.values()
        )
        return loaded, actual_bytes

    def ensure(self, client_ids):
        protected = set(client_ids)
        for client_id in client_ids:
            if client_id in self.entries:
                self.hits += 1
                self.entries.move_to_end(client_id)
                continue
            self.misses += 1
            needed = self.client_bytes[client_id]
            while self.current_bytes + needed > self.cache_budget_bytes:
                eviction_id = next(
                    (
                        candidate
                        for candidate in self.entries
                        if candidate not in protected
                    ),
                    None,
                )
                if eviction_id is None:
                    raise RuntimeError(
                        f"GPU {self.gpu_id} cache cannot hold active clients "
                        f"{sorted(protected)} within budget"
                    )
                entry = self.entries.pop(eviction_id)
                self.current_bytes -= entry["actual_bytes"]
                self.evictions += 1
                del entry
                torch.cuda.empty_cache()
            tensors, actual_bytes = self._load_client(client_id)
            self.entries[client_id] = {
                **tensors,
                "actual_bytes": actual_bytes,
            }
            self.current_bytes += actual_bytes
            self.logger.info(
                "Cached client %d on GPU %d: %.2f MiB",
                client_id,
                self.gpu_id,
                actual_bytes / MIB,
            )

    def get(self, client_id):
        self.ensure([client_id])
        return self.entries[client_id]

    def clear(self):
        self.entries.clear()
        self.current_bytes = 0
        torch.cuda.empty_cache()

    def stats(self):
        return {
            "gpu_id": self.gpu_id,
            "mode": self.mode,
            "cache_budget_bytes": self.cache_budget_bytes,
            "cache_budget_mib": self.cache_budget_bytes / MIB,
            "planned_bytes": self.planned_bytes,
            "planned_mib": self.planned_bytes / MIB,
            "current_bytes": self.current_bytes,
            "current_mib": self.current_bytes / MIB,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "assigned_clients": list(self.assigned_clients),
        }


def make_benchmark_pair(family, device, stream):
    personal = build_model(family).to(device)
    proxy = build_model("cnn1d").to(device)
    optimizer = torch.optim.Adam(
        list(personal.parameters()) + list(proxy.parameters()),
        lr=CONFIG["learning_rate"],
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(CONFIG["seed"] + 700 + len(family))
    features = torch.randn(
        CONFIG["per_client_batch_size"],
        CONFIG["num_features"],
        generator=generator,
        device=device,
    )
    targets = torch.randint(
        0,
        CONFIG["num_classes"],
        (CONFIG["per_client_batch_size"],),
        generator=generator,
        device=device,
    )
    return {
        "family": family,
        "personal": personal,
        "proxy": proxy,
        "optimizer": optimizer,
        "scaler": scaler,
        "features": features,
        "targets": targets,
        "stream": stream,
    }


def enqueue_benchmark_step(context):
    with torch.cuda.stream(context["stream"]):
        context["optimizer"].zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            personal_logits = context["personal"](context["features"])
            proxy_logits = context["proxy"](context["features"])
            personal_loss = F.cross_entropy(personal_logits, context["targets"])
            proxy_loss = F.cross_entropy(proxy_logits, context["targets"])
            discrepancy = F.mse_loss(
                personal_logits.softmax(dim=1),
                proxy_logits.softmax(dim=1),
            )
            adaptive = discrepancy / (
                personal_loss.detach()
                + proxy_loss.detach()
                + CONFIG["adaptive_epsilon"]
            )
            loss = personal_loss + proxy_loss + adaptive
        context["scaler"].scale(loss).backward()
        context["scaler"].step(context["optimizer"])
        context["scaler"].update()


def benchmark_single_family(family, device):
    stream = torch.cuda.Stream(device=device)
    context = make_benchmark_pair(family, device, stream)
    for _ in range(CONFIG["benchmark_warmup_steps"]):
        enqueue_benchmark_step(context)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
    for _ in range(CONFIG["benchmark_timed_steps"]):
        enqueue_benchmark_step(context)
    with torch.cuda.stream(stream):
        end.record()
    stream.synchronize()
    seconds = start.elapsed_time(end) / 1000.0
    del context
    torch.cuda.empty_cache()
    return seconds / CONFIG["benchmark_timed_steps"]


def benchmark_stream_candidate(families, stream_count, device):
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    contexts = [
        make_benchmark_pair(
            family,
            device,
            streams[index % stream_count],
        )
        for index, family in enumerate(families)
    ]
    for _ in range(CONFIG["stream_benchmark_warmup_steps"]):
        for context in contexts:
            enqueue_benchmark_step(context)
    for stream in streams:
        stream.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in streams]
    ends = [torch.cuda.Event(enable_timing=True) for _ in streams]
    for stream, event in zip(streams, starts):
        with torch.cuda.stream(stream):
            event.record()
    for _ in range(CONFIG["stream_benchmark_timed_steps"]):
        for context in contexts:
            enqueue_benchmark_step(context)
    for stream, event in zip(streams, ends):
        with torch.cuda.stream(stream):
            event.record()
    for stream in streams:
        stream.synchronize()
    elapsed_seconds = max(
        start.elapsed_time(end) / 1000.0
        for start, end in zip(starts, ends)
    )
    total_examples = (
        len(contexts)
        * CONFIG["stream_benchmark_timed_steps"]
        * CONFIG["per_client_batch_size"]
    )
    result = {
        "stream_count": stream_count,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_examples / max(elapsed_seconds, 1e-12),
        "families": list(families),
    }
    del contexts
    torch.cuda.empty_cache()
    return result


def choose_stream_count(assigned_clients, device):
    ordered_families = [
        CONFIG["client_architectures"][client_id - 1]
        for client_id in assigned_clients
    ]
    unique_families = list(dict.fromkeys(ordered_families))
    benchmark_families = (
        unique_families[:2]
        if len(unique_families) >= 2
        else unique_families * 2
    )
    results = {
        stream_count: benchmark_stream_candidate(
            benchmark_families,
            stream_count,
            device,
        )
        for stream_count in CONFIG["stream_candidates"]
    }
    baseline = results[1]["samples_per_second"]
    candidate = results.get(2, results[1])["samples_per_second"]
    speedup = candidate / max(baseline, 1e-12)
    selected = 2 if 2 in results and speedup >= CONFIG["stream_min_speedup"] else 1
    return {
        "selected_stream_count": selected,
        "two_stream_speedup": speedup,
        "results": results,
    }


def evaluate_gpu_indices(model, client_data, index_tensor, device):
    model.eval()
    matrix = torch.zeros(
        (CONFIG["num_classes"], CONFIG["num_classes"]),
        dtype=torch.int64,
        device=device,
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    examples = len(index_tensor)
    with torch.no_grad():
        for start in range(0, examples, CONFIG["per_client_batch_size"]):
            end = min(examples, start + CONFIG["per_client_batch_size"])
            raw_indices = index_tensor[start:end]
            features = client_data["features"].index_select(0, raw_indices)
            targets = client_data["labels"].index_select(0, raw_indices)
            with torch.amp.autocast("cuda"):
                logits = model(features)
                loss = F.cross_entropy(logits, targets)
            predictions = logits.argmax(dim=1)
            batch_examples = end - start
            loss_sum += loss.detach().to(torch.float64) * batch_examples
            encoded = targets.to(torch.int64) * CONFIG["num_classes"] + predictions
            matrix += torch.bincount(
                encoded,
                minlength=CONFIG["num_classes"] ** 2,
            ).reshape(CONFIG["num_classes"], CONFIG["num_classes"])
    matrix_np = matrix.cpu().numpy()
    loss_value = float(loss_sum.cpu().item() / max(examples, 1))
    return {
        "loss": loss_value,
        "examples": examples,
        "confusion_matrix": matrix_np,
        "metrics": metrics_from_confusion(matrix_np),
    }


def make_training_context(
    client_id,
    stream,
    device,
    personal_model,
    proxy_model,
    data,
    round_index,
    phase,
):
    personal_model.train()
    if proxy_model is not None:
        proxy_model.train()
        parameters = list(personal_model.parameters()) + list(proxy_model.parameters())
    else:
        parameters = list(personal_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=CONFIG["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_seed(round_index, client_id, phase))
    loss_width = 3 if proxy_model is not None else 1
    # The caller makes this client stream wait for prior default-stream model
    # setup. Create every CUDA tensor owned by the context on that stream:
    # wait_stream() only orders work enqueued before the call, so creating the
    # permutation later on the default stream would race with index_select().
    with torch.cuda.stream(stream):
        order = torch.randperm(
            len(data["train_indices"]),
            generator=generator,
            device=device,
        )
        loss_sums = torch.zeros(
            loss_width,
            dtype=torch.float64,
            device=device,
        )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    return {
        "client_id": client_id,
        "stream": stream,
        "personal": personal_model,
        "proxy": proxy_model,
        "data": data,
        "optimizer": optimizer,
        "scaler": scaler,
        "order": order,
        "offset": 0,
        "processed": 0,
        "optimizer_steps": 0,
        "loss_sums": loss_sums,
        "start_event": start_event,
        "end_event": end_event,
        "started": False,
        "finished": False,
        "phase": phase,
    }


def enqueue_training_step(context):
    start = context["offset"]
    end = min(
        len(context["order"]),
        start + CONFIG["per_client_batch_size"],
    )
    with torch.cuda.stream(context["stream"]):
        if not context["started"]:
            context["start_event"].record()
            context["started"] = True
        order_slice = context["order"][start:end]
        raw_indices = context["data"]["train_indices"].index_select(0, order_slice)
        features = context["data"]["features"].index_select(0, raw_indices)
        targets = context["data"]["labels"].index_select(0, raw_indices)
        context["optimizer"].zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            personal_logits = context["personal"](features)
            personal_loss = F.cross_entropy(personal_logits, targets)
            if context["proxy"] is not None:
                proxy_logits = context["proxy"](features)
                proxy_loss = F.cross_entropy(proxy_logits, targets)
                discrepancy = F.mse_loss(
                    personal_logits.softmax(dim=1),
                    proxy_logits.softmax(dim=1),
                )
                distillation_loss = discrepancy / (
                    personal_loss.detach()
                    + proxy_loss.detach()
                    + CONFIG["adaptive_epsilon"]
                )
                joint_loss = personal_loss + proxy_loss + distillation_loss
            else:
                proxy_loss = None
                distillation_loss = None
                joint_loss = personal_loss
        context["scaler"].scale(joint_loss).backward()
        context["scaler"].step(context["optimizer"])
        context["scaler"].update()
        batch_examples = end - start
        if context["proxy"] is not None:
            context["loss_sums"] += torch.stack(
                [
                    personal_loss.detach().to(torch.float64),
                    proxy_loss.detach().to(torch.float64),
                    distillation_loss.detach().to(torch.float64),
                ]
            ) * batch_examples
        else:
            context["loss_sums"][0] += (
                personal_loss.detach().to(torch.float64) * batch_examples
            )
        context["offset"] = end
        context["processed"] += batch_examples
        context["optimizer_steps"] += 1
        if end == len(context["order"]):
            context["end_event"].record()
            context["finished"] = True


def train_owned_clients(
    assigned_clients,
    work_order,
    stream_count,
    cache,
    personal_models,
    proxy_models,
    global_proxy_state,
    round_index,
    device,
    phase,
):
    pending = list(work_order)
    stream_pool = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    free_streams = list(stream_pool)
    active = []
    completed = []
    default_stream = torch.cuda.current_stream(device)
    wall_started = time.perf_counter()

    if phase == "mutual":
        for client_id in assigned_clients:
            proxy_models[client_id].load_state_dict(global_proxy_state, strict=True)

    while pending or active:
        while pending and free_streams:
            client_id = pending.pop(0)
            protected_clients = [
                context["client_id"] for context in active
            ] + [client_id]
            cache.ensure(protected_clients)
            data = cache.entries[client_id]
            cache.entries.move_to_end(client_id)
            stream = free_streams.pop(0)
            stream.wait_stream(default_stream)
            context = make_training_context(
                client_id,
                stream,
                device,
                personal_models[client_id],
                proxy_models[client_id] if phase == "mutual" else None,
                data,
                round_index,
                phase,
            )
            active.append(context)
        finished_now = []
        for context in active:
            enqueue_training_step(context)
            if context["finished"]:
                finished_now.append(context)
        for context in finished_now:
            if cache.mode == "deterministic_lru_gpu_cache":
                context["stream"].synchronize()
                context["data"] = None
            active.remove(context)
            completed.append(context)
            free_streams.append(context["stream"])

    for stream in stream_pool:
        default_stream.wait_stream(stream)
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started

    results = {}
    for context in completed:
        client_id = context["client_id"]
        losses = (
            context["loss_sums"].cpu().numpy()
            / max(context["processed"], 1)
        ).tolist()
        elapsed_seconds = (
            context["start_event"].elapsed_time(context["end_event"]) / 1000.0
        )
        result = {
            "client_id": client_id,
            "processed_examples_with_padding": context["processed"],
            "optimizer_steps": context["optimizer_steps"],
            "sampler_padding_rows": 0,
            "train_seconds": elapsed_seconds,
            "worker_phase_wall_seconds": wall_seconds,
        }
        if phase == "mutual":
            result.update(
                {
                    "personalized_label_loss": float(losses[0]),
                    "proxy_label_loss": float(losses[1]),
                    "adaptive_distillation_loss": float(losses[2]),
                    "personalized_total_loss": float(losses[0] + losses[2]),
                    "proxy_total_loss": float(losses[1] + losses[2]),
                }
            )
        else:
            result.update(
                {
                    "personalized_label_loss": float(losses[0]),
                    "proxy_label_loss": 0.0,
                    "adaptive_distillation_loss": 0.0,
                    "personalized_total_loss": float(losses[0]),
                    "proxy_total_loss": 0.0,
                }
            )
        results[client_id] = result
    return results, wall_seconds


def load_test_to_gpu(device):
    validate_numpy_classification_cache(
        DATA["test"]["features_path"],
        DATA["test"]["labels_path"],
        "global test",
    )
    copy_stream = torch.cuda.Stream(device=device)
    features = load_numpy_to_device(
        DATA["test"]["features_path"],
        device,
        CONFIG["gpu_cache_chunk_rows"],
        copy_stream,
    )
    labels = load_numpy_to_device(
        DATA["test"]["labels_path"],
        device,
        CONFIG["gpu_cache_chunk_rows"],
        copy_stream,
    )
    return {"features": features, "labels": labels}


def evaluate_test_range(model, test_data, start_index, end_index, device):
    model.eval()
    matrix = torch.zeros(
        (CONFIG["num_classes"], CONFIG["num_classes"]),
        dtype=torch.int64,
        device=device,
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(
            start_index,
            end_index,
            CONFIG["per_client_batch_size"],
        ):
            end = min(end_index, start + CONFIG["per_client_batch_size"])
            features = test_data["features"][start:end]
            targets = test_data["labels"][start:end]
            with torch.amp.autocast("cuda"):
                logits = model(features)
                loss = F.cross_entropy(logits, targets)
            predictions = logits.argmax(dim=1)
            batch_examples = end - start
            loss_sum += loss.detach().to(torch.float64) * batch_examples
            encoded = targets.to(torch.int64) * CONFIG["num_classes"] + predictions
            matrix += torch.bincount(
                encoded,
                minlength=CONFIG["num_classes"] ** 2,
            ).reshape(CONFIG["num_classes"], CONFIG["num_classes"])
    examples = end_index - start_index
    return {
        "loss_sum": float(loss_sum.cpu().item()),
        "examples": examples,
        "confusion_matrix": matrix.cpu().numpy(),
    }


def worker_main(gpu_id, task_queue, result_queue, manifest_path):
    worker_log_path = CACHE_DIR / f"gpu_{gpu_id}_worker.log"
    logger = setup_logger(worker_log_path, f"gpu_worker_{gpu_id}")
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
        torch.set_num_threads(CONFIG["omp_num_threads_per_process"])
        random.seed(CONFIG["seed"] + gpu_id)
        np.random.seed(CONFIG["seed"] + gpu_id)
        torch.manual_seed(CONFIG["seed"] + gpu_id)
        torch.cuda.manual_seed_all(CONFIG["seed"] + gpu_id)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Worker %d started on %s", gpu_id, torch.cuda.get_device_name(gpu_id))

        assigned_clients = ()
        work_order = ()
        cache = None
        personal_models = {}
        proxy_models = {}
        selected_stream_count = 1
        initial_states = None

        while True:
            task = task_queue.get()
            task_id = task["task_id"]
            task_type = task["task_type"]
            if task_type == "shutdown":
                result_queue.put(
                    {
                        "kind": "shutdown_ack",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                    }
                )
                break
            if task_type == "benchmark":
                families = task["families"]
                seconds_per_step = {
                    family: benchmark_single_family(family, device)
                    for family in families
                }
                properties = torch.cuda.get_device_properties(gpu_id)
                result_queue.put(
                    {
                        "kind": "benchmark_result",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "seconds_per_step": seconds_per_step,
                        "environment": {
                            "gpu_id": gpu_id,
                            "gpu_name": properties.name,
                            "vram_bytes": properties.total_memory,
                            "compute_capability": [
                                properties.major,
                                properties.minor,
                            ],
                            "torch_version": torch.__version__,
                            "cuda_version": torch.version.cuda,
                        },
                    }
                )
                continue
            if task_type == "initialize_assignment":
                assigned_clients = tuple(task["assigned_clients"])
                work_order = tuple(task["work_order"])
                initial_states = torch.load(
                    task["initial_states_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                cache = GpuClientCache(gpu_id, assigned_clients, device, logger)
                for client_id in assigned_clients:
                    family = CONFIG["client_architectures"][client_id - 1]
                    personal = build_model(family).to(device)
                    personal.load_state_dict(initial_states[family], strict=True)
                    personal_models[client_id] = personal
                    proxy = build_model("cnn1d").to(device)
                    proxy.load_state_dict(initial_states["cnn1d"], strict=True)
                    proxy_models[client_id] = proxy
                stream_benchmark = choose_stream_count(assigned_clients, device)
                selected_stream_count = stream_benchmark["selected_stream_count"]
                if selected_stream_count == 2:
                    largest_two_clients = sorted(
                        cache.client_bytes.values(),
                        reverse=True,
                    )[:2]
                    if sum(largest_two_clients) > cache.cache_budget_bytes:
                        selected_stream_count = 1
                        stream_benchmark["selected_stream_count"] = 1
                        stream_benchmark["memory_safety_override"] = (
                            "Two simultaneously active client caches exceed "
                            "the configured GPU cache budget."
                        )
                result_queue.put(
                    {
                        "kind": "initialize_result",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "cache": cache.stats(),
                        "stream_benchmark": stream_benchmark,
                        "selected_stream_count": selected_stream_count,
                    }
                )
                continue
            if task_type == "train_round":
                torch.cuda.reset_peak_memory_stats(device)
                global_proxy_state = torch.load(
                    task["global_proxy_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                results, worker_wall = train_owned_clients(
                    assigned_clients,
                    work_order,
                    selected_stream_count,
                    cache,
                    personal_models,
                    proxy_models,
                    global_proxy_state,
                    task["round"],
                    device,
                    "mutual",
                )
                for client_id in assigned_clients:
                    validation_started = time.perf_counter()
                    data = cache.get(client_id)
                    evaluation = evaluate_gpu_indices(
                        personal_models[client_id],
                        data,
                        data["val_indices"],
                        device,
                    )
                    results[client_id]["personalized_validation"] = evaluation
                    results[client_id]["validation_seconds"] = (
                        time.perf_counter() - validation_started
                    )
                payload = {
                    "clients": {
                        str(client_id): {
                            "personalized_state": cpu_state(
                                personal_models[client_id]
                            ),
                            "proxy_state": cpu_state(proxy_models[client_id]),
                            "result": results[client_id],
                        }
                        for client_id in assigned_clients
                    }
                }
                payload_path = (
                    CACHE_DIR
                    / f"gpu_{gpu_id}_round_{task['round']}_train_payload.pt"
                )
                torch.save(payload, payload_path)
                result_queue.put(
                    {
                        "kind": "train_round_result",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                        "payload_bytes": payload_path.stat().st_size,
                        "worker_wall_seconds": worker_wall,
                        "cache": cache.stats(),
                        "memory": {
                            "allocated_bytes": torch.cuda.memory_allocated(device),
                            "reserved_bytes": torch.cuda.memory_reserved(device),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                                device
                            ),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                                device
                            ),
                        },
                    }
                )
                continue
            if task_type == "validate_proxy":
                global_proxy_state = torch.load(
                    task["global_proxy_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                proxy = build_model("cnn1d").to(device)
                proxy.load_state_dict(global_proxy_state, strict=True)
                evaluations = {}
                for client_id in assigned_clients:
                    started = time.perf_counter()
                    data = cache.get(client_id)
                    evaluation = evaluate_gpu_indices(
                        proxy,
                        data,
                        data["val_indices"],
                        device,
                    )
                    evaluation["seconds"] = time.perf_counter() - started
                    evaluations[str(client_id)] = evaluation
                payload_path = (
                    CACHE_DIR
                    / f"gpu_{gpu_id}_round_{task['round']}_proxy_validation.pt"
                )
                torch.save(evaluations, payload_path)
                result_queue.put(
                    {
                        "kind": "validate_proxy_result",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                        "payload_bytes": payload_path.stat().st_size,
                    }
                )
                del proxy
                torch.cuda.empty_cache()
                continue
            if task_type == "finetune":
                checkpoint = torch.load(
                    task["checkpoint_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                checkpoint_states = checkpoint["personalized_model_state_dicts"]
                for client_id in assigned_clients:
                    personal_models[client_id].load_state_dict(
                        checkpoint_states[str(client_id)],
                        strict=True,
                    )
                results, worker_wall = train_owned_clients(
                    assigned_clients,
                    work_order,
                    selected_stream_count,
                    cache,
                    personal_models,
                    proxy_models,
                    None,
                    CONFIG["communication_rounds"] + 1,
                    device,
                    "finetune",
                )
                payload = {
                    "clients": {
                        str(client_id): {
                            "personalized_state": cpu_state(
                                personal_models[client_id]
                            ),
                            "result": results[client_id],
                        }
                        for client_id in assigned_clients
                    }
                }
                payload_path = CACHE_DIR / f"gpu_{gpu_id}_finetune_payload.pt"
                torch.save(payload, payload_path)
                result_queue.put(
                    {
                        "kind": "finetune_result",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                        "payload_bytes": payload_path.stat().st_size,
                        "worker_wall_seconds": worker_wall,
                    }
                )
                continue
            if task_type == "final_evaluation":
                cache.clear()
                del personal_models
                del proxy_models
                torch.cuda.empty_cache()
                test_load_started = time.perf_counter()
                test_data = load_test_to_gpu(device)
                test_cache_seconds = time.perf_counter() - test_load_started
                checkpoint = torch.load(
                    task["checkpoint_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                proxy = build_model("cnn1d").to(device)
                proxy.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=True,
                )
                shard = task["proxy_test_shard"]
                proxy_evaluation = evaluate_test_range(
                    proxy,
                    test_data,
                    shard[0],
                    shard[1],
                    device,
                )
                personalized = {}
                for client_id in task["evaluation_clients"]:
                    family = CONFIG["client_architectures"][client_id - 1]
                    model = build_model(family).to(device)
                    model.load_state_dict(
                        checkpoint["personalized_model_state_dicts"][
                            str(client_id)
                        ],
                        strict=True,
                    )
                    evaluation = evaluate_test_range(
                        model,
                        test_data,
                        0,
                        DATA["test"]["examples"],
                        device,
                    )
                    matrix = evaluation["confusion_matrix"]
                    personalized[str(client_id)] = {
                        "architecture": family,
                        "loss": evaluation["loss_sum"]
                        / max(evaluation["examples"], 1),
                        **metrics_from_confusion(matrix),
                    }
                    del model
                payload = {
                    "proxy_shard": proxy_evaluation,
                    "personalized": personalized,
                    "test_cache_seconds": test_cache_seconds,
                    "test_cache_bytes": (
                        test_data["features"].numel()
                        * test_data["features"].element_size()
                        + test_data["labels"].numel()
                        * test_data["labels"].element_size()
                    ),
                }
                payload_path = CACHE_DIR / f"gpu_{gpu_id}_final_evaluation.pt"
                torch.save(payload, payload_path)
                result_queue.put(
                    {
                        "kind": "final_evaluation_result",
                        "task_id": task_id,
                        "gpu_id": gpu_id,
                        "payload_path": str(payload_path),
                        "payload_bytes": payload_path.stat().st_size,
                        "memory": {
                            "allocated_bytes": torch.cuda.memory_allocated(device),
                            "reserved_bytes": torch.cuda.memory_reserved(device),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                                device
                            ),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                                device
                            ),
                        },
                    }
                )
                continue
            raise ValueError(f"Unknown task type: {task_type}")
    except Exception:
        logger.error("GPU worker failed:\n%s", traceback.format_exc())
        result_queue.put(
            {
                "kind": "worker_error",
                "gpu_id": gpu_id,
                "traceback": traceback.format_exc(),
            }
        )
        raise


class WorkerPool:
    def __init__(self, logger):
        self.logger = logger
        self.context = multiprocessing.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.task_queues = [self.context.Queue() for _ in range(2)]
        self.workers = [
            self.context.Process(
                target=worker_main,
                args=(
                    gpu_id,
                    self.task_queues[gpu_id],
                    self.result_queue,
                    str(MANIFEST_PATH),
                ),
                name=f"gpu-worker-{gpu_id}",
            )
            for gpu_id in range(2)
        ]
        self.task_sequence = 0

    def start(self):
        for worker in self.workers:
            worker.start()

    def submit(self, gpu_id, task_type, **payload):
        self.task_sequence += 1
        task_id = f"{task_type}-gpu{gpu_id}-{self.task_sequence}"
        self.task_queues[gpu_id].put(
            {
                "task_id": task_id,
                "task_type": task_type,
                **payload,
            }
        )
        return task_id

    def submit_both(self, task_type, payload_by_gpu=None, **shared):
        payload_by_gpu = payload_by_gpu or {}
        return {
            gpu_id: self.submit(
                gpu_id,
                task_type,
                **shared,
                **payload_by_gpu.get(gpu_id, {}),
            )
            for gpu_id in range(2)
        }

    def collect(self, task_ids):
        expected = set(task_ids.values())
        results = {}
        while expected:
            try:
                message = self.result_queue.get(
                    timeout=CONFIG["worker_health_poll_seconds"]
                )
            except queue.Empty:
                dead = [
                    (index, worker.exitcode)
                    for index, worker in enumerate(self.workers)
                    if not worker.is_alive()
                ]
                if dead:
                    raise RuntimeError(
                        f"GPU worker exited before completing tasks: {dead}"
                    )
                continue
            if message.get("kind") == "worker_error":
                raise RuntimeError(
                    f"GPU worker {message.get('gpu_id')} failed:\n"
                    f"{message.get('traceback')}"
                )
            task_id = message.get("task_id")
            if task_id not in expected:
                raise RuntimeError(f"Unexpected or duplicate worker result: {task_id}")
            expected.remove(task_id)
            results[int(message["gpu_id"])] = message
        return results

    def shutdown(self):
        task_ids = self.submit_both("shutdown")
        try:
            self.collect(task_ids)
        finally:
            for worker in self.workers:
                worker.join(timeout=CONFIG["worker_join_timeout_seconds"])
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            failures = [
                (worker.name, worker.exitcode)
                for worker in self.workers
                if worker.exitcode != 0
            ]
            if failures:
                raise RuntimeError(f"GPU workers had nonzero exit codes: {failures}")


class GpuUtilizationMonitor:
    def __init__(self):
        self.samples = {0: [], 1: []}
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
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
                    timeout=5,
                )
                timestamp = time.time()
                for line in completed.stdout.strip().splitlines():
                    gpu_text, utilization_text, memory_text = [
                        value.strip() for value in line.split(",")
                    ]
                    gpu_id = int(gpu_text)
                    if gpu_id in self.samples:
                        self.samples[gpu_id].append(
                            {
                                "timestamp": timestamp,
                                "utilization_percent": float(utilization_text),
                                "memory_used_mib": float(memory_text),
                            }
                        )
            except Exception as error:
                self.errors.append(str(error))
            self.stop_event.wait(CONFIG["gpu_monitor_interval_seconds"])

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def summary(self):
        output = {"errors": self.errors[:10], "gpus": {}}
        for gpu_id, samples in self.samples.items():
            utilization = [item["utilization_percent"] for item in samples]
            memory = [item["memory_used_mib"] for item in samples]
            output["gpus"][str(gpu_id)] = {
                "sample_count": len(samples),
                "mean_utilization_percent": (
                    float(np.mean(utilization)) if utilization else None
                ),
                "p95_utilization_percent": (
                    float(np.percentile(utilization, 95)) if utilization else None
                ),
                "max_utilization_percent": max(utilization) if utilization else None,
                "mean_memory_used_mib": (
                    float(np.mean(memory)) if memory else None
                ),
                "max_memory_used_mib": max(memory) if memory else None,
            }
        return output


def exact_client_assignment(benchmark_by_gpu):
    clients = tuple(range(1, CONFIG["num_clients"] + 1))
    best = None
    for mask in range(1, 2 ** len(clients) - 1):
        gpu_0_clients = tuple(
            client_id
            for index, client_id in enumerate(clients)
            if mask & (1 << index)
        )
        gpu_1_clients = tuple(
            client_id for client_id in clients if client_id not in gpu_0_clients
        )
        load_0 = sum(
            math.ceil(
                DATA["clients"][str(client_id)]["train_examples"]
                / CONFIG["per_client_batch_size"]
            )
            * benchmark_by_gpu[0][
                CONFIG["client_architectures"][client_id - 1]
            ]
            for client_id in gpu_0_clients
        )
        load_1 = sum(
            math.ceil(
                DATA["clients"][str(client_id)]["train_examples"]
                / CONFIG["per_client_batch_size"]
            )
            * benchmark_by_gpu[1][
                CONFIG["client_architectures"][client_id - 1]
            ]
            for client_id in gpu_1_clients
        )
        objective = (
            max(load_0, load_1),
            abs(load_0 - load_1),
            gpu_0_clients,
        )
        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "client_assignment": {
                    0: gpu_0_clients,
                    1: gpu_1_clients,
                },
                "predicted_load_seconds": {
                    0: load_0,
                    1: load_1,
                },
            }
    return best


def work_order_for_gpu(client_ids, gpu_id, benchmark_by_gpu):
    return tuple(
        sorted(
            client_ids,
            key=lambda client_id: (
                -math.ceil(
                    DATA["clients"][str(client_id)]["train_examples"]
                    / CONFIG["per_client_batch_size"]
                )
                * benchmark_by_gpu[gpu_id][
                    CONFIG["client_architectures"][client_id - 1]
                ],
                client_id,
            ),
        )
    )


def exact_evaluation_assignment(benchmark_by_gpu):
    clients = tuple(range(1, CONFIG["num_clients"] + 1))
    test_steps = math.ceil(
        DATA["test"]["examples"] / CONFIG["per_client_batch_size"]
    )
    best = None
    for mask in range(1, 2 ** len(clients) - 1):
        groups = {
            0: tuple(
                client_id
                for index, client_id in enumerate(clients)
                if mask & (1 << index)
            )
        }
        groups[1] = tuple(
            client_id for client_id in clients if client_id not in groups[0]
        )
        loads = {
            gpu_id: sum(
                test_steps
                * benchmark_by_gpu[gpu_id][
                    CONFIG["client_architectures"][client_id - 1]
                ]
                for client_id in groups[gpu_id]
            )
            for gpu_id in range(2)
        }
        objective = (max(loads.values()), abs(loads[0] - loads[1]), groups[0])
        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "assignment": groups,
                "predicted_load_seconds": loads,
            }
    return best


def combine_evaluations(evaluations):
    matrix = np.zeros(
        (CONFIG["num_classes"], CONFIG["num_classes"]),
        dtype=np.int64,
    )
    loss_sum = 0.0
    examples = 0
    for evaluation in evaluations:
        matrix += np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
        if "loss_sum" in evaluation:
            loss_sum += evaluation["loss_sum"]
        else:
            loss_sum += evaluation["loss"] * evaluation["examples"]
        examples += evaluation["examples"]
    return {
        "loss": loss_sum / max(examples, 1),
        "examples": examples,
        "confusion_matrix": matrix,
        "metrics": metrics_from_confusion(matrix),
    }


def create_plots(history_round, class_matrix, classification_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = OUT / "artifacts"
    distribution = pd.read_csv(OUT / "metrics" / "client_class_distribution.csv")
    totals = distribution.groupby("class_name")["train_examples"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(12, 9))
    totals.plot.barh(ax=ax)
    ax.set_title("Training class distribution across all clients")
    ax.set_xlabel("Examples")
    fig.tight_layout()
    fig.savefig(artifacts / "class_distribution.png", dpi=160)
    plt.close(fig)

    rounds = [row["round"] for row in history_round]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        rounds,
        [row["global_validation_accuracy"] for row in history_round],
        label="accuracy",
    )
    ax.plot(
        rounds,
        [row["global_validation_macro_f1"] for row in history_round],
        label="macro-F1",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(artifacts / "accuracy_f1_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        rounds,
        [row["hard_loss"] for row in history_round],
        label="personalized CE",
    )
    ax.plot(
        rounds,
        [row["proxy_label_loss"] for row in history_round],
        label="proxy CE",
    )
    ax.plot(
        rounds,
        [row["soft_loss"] for row in history_round],
        label="adaptive distillation",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(artifacts / "loss_curves.png", dpi=160)
    plt.close(fig)

    normalized = class_matrix.astype(np.float64)
    normalized = safe_divide(normalized, normalized.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Row-normalized proxy confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(artifacts / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    per_class = [
        row for row in classification_rows if isinstance(row["class_id"], int)
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(
        [row["class_name"] for row in per_class],
        [row["f1"] for row in per_class],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Proxy per-class F1")
    fig.tight_layout()
    fig.savefig(artifacts / "per_class_f1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(rounds, [row["round_seconds"] for row in history_round])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime per round")
    fig.tight_layout()
    fig.savefig(artifacts / "runtime_per_round.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        rounds,
        [row["communication_mib_cumulative"] for row in history_round],
        marker="o",
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative logical FL communication (MiB)")
    fig.tight_layout()
    fig.savefig(artifacts / "communication_cumulative.png", dpi=160)
    plt.close(fig)


def merge_worker_logs():
    run_log = OUT / "logs" / "run.log"
    gpu_0_log = CACHE_DIR / "gpu_0_worker.log"
    gpu_1_log = CACHE_DIR / "gpu_1_worker.log"
    if gpu_0_log.is_file():
        with run_log.open("a", encoding="utf-8") as destination:
            destination.write("\n--- GPU 0 worker diagnostics ---\n")
            destination.write(gpu_0_log.read_text(encoding="utf-8"))
    rank_1_log = OUT / "logs" / "rank_1.log"
    if gpu_1_log.is_file():
        rank_1_log.write_text(
            gpu_1_log.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    else:
        rank_1_log.write_text(
            "GPU 1 worker log was not produced.\n",
            encoding="utf-8",
        )


def coordinator_run(logger):
    runtime_started = time.perf_counter()
    required_families = sorted(set(CONFIG["client_architectures"] + ["cnn1d"]))
    initial_states = {
        family: make_initial_state(family) for family in required_families
    }
    repeated_states = {
        family: make_initial_state(family) for family in required_families
    }
    initialization_hashes = {
        family: state_sha256(state) for family, state in initial_states.items()
    }
    if initialization_hashes != {
        family: state_sha256(state) for family, state in repeated_states.items()
    }:
        raise RuntimeError("Deterministic initialization contract failed")
    expected_parameters = {"gru": 40034, "transformer": 71010, "cnn1d": 35874}
    model_metadata_by_family = {
        family: model_metadata(family, initial_states[family])
        for family in required_families
    }
    for family, metadata in model_metadata_by_family.items():
        if metadata["trainable_parameters"] != expected_parameters[family]:
            raise RuntimeError(
                f"{family} parameter count {metadata['trainable_parameters']} "
                f"!= {expected_parameters[family]}"
            )
    initial_states_path = CACHE_DIR / "initial_model_states.pt"
    torch.save(initial_states, initial_states_path)
    config_path = OUT / "metrics" / "config.json"
    config_output = json.loads(config_path.read_text(encoding="utf-8"))
    config_output["initialization_hashes"] = initialization_hashes
    config_output["model_metadata_by_family"] = model_metadata_by_family
    write_json(config_path, config_output)

    pool = WorkerPool(logger)
    monitor = GpuUtilizationMonitor()
    interprocess_payload_bytes = 0
    pool.start()
    monitor.start()
    environment_by_worker = []
    worker_benchmarks = {}
    initialization_results = {}
    history_local_epoch = []
    history_client = []
    history_round = []
    best_metric = -math.inf
    best_round = 0
    cumulative_training_seconds = 0.0
    cumulative_communication_bytes = 0
    total_local_train_seconds = 0.0
    total_personal_validation_seconds = 0.0
    total_global_validation_seconds = 0.0
    round_worker_wall = {}
    fine_tune_results = {}
    run_failed = False

    try:
        active_personal_families = sorted(set(CONFIG["client_architectures"]))
        benchmark_task_ids = pool.submit_both(
            "benchmark",
            families=active_personal_families,
        )
        benchmark_results = pool.collect(benchmark_task_ids)
        benchmark_by_gpu = {
            gpu_id: result["seconds_per_step"]
            for gpu_id, result in benchmark_results.items()
        }
        worker_benchmarks = benchmark_results
        environment_by_worker = [
            benchmark_results[gpu_id]["environment"] for gpu_id in range(2)
        ]
        assignment_plan = exact_client_assignment(benchmark_by_gpu)
        client_assignment = assignment_plan["client_assignment"]
        work_order = {
            gpu_id: work_order_for_gpu(
                client_assignment[gpu_id],
                gpu_id,
                benchmark_by_gpu,
            )
            for gpu_id in range(2)
        }
        logger.info(
            "Client assignment: GPU0=%s GPU1=%s predicted seconds=%s",
            client_assignment[0],
            client_assignment[1],
            assignment_plan["predicted_load_seconds"],
        )
        initialize_task_ids = pool.submit_both(
            "initialize_assignment",
            payload_by_gpu={
                gpu_id: {
                    "assigned_clients": list(client_assignment[gpu_id]),
                    "work_order": list(work_order[gpu_id]),
                }
                for gpu_id in range(2)
            },
            initial_states_path=str(initial_states_path),
        )
        initialization_results = pool.collect(initialize_task_ids)

        global_proxy_state = clone_state(initial_states["cnn1d"])
        personalized_states = {
            client_id: clone_state(
                initial_states[CONFIG["client_architectures"][client_id - 1]]
            )
            for client_id in range(1, CONFIG["num_clients"] + 1)
        }
        proxy_state_bytes = model_metadata_by_family["cnn1d"]["model_state_bytes"]
        fl_bytes_per_round = 2 * CONFIG["num_clients"] * proxy_state_bytes
        training_started = time.perf_counter()

        for round_index in range(1, CONFIG["communication_rounds"] + 1):
            round_started = time.perf_counter()
            proxy_path = CACHE_DIR / f"round_{round_index}_global_proxy.pt"
            torch.save(global_proxy_state, proxy_path)
            train_task_ids = pool.submit_both(
                "train_round",
                global_proxy_path=str(proxy_path),
                round=round_index,
            )
            train_results = pool.collect(train_task_ids)
            round_worker_wall[str(round_index)] = {
                str(gpu_id): result["worker_wall_seconds"]
                for gpu_id, result in train_results.items()
            }
            client_train_payload = {}
            for gpu_id, worker_result in train_results.items():
                interprocess_payload_bytes += worker_result["payload_bytes"]
                payload = torch.load(
                    worker_result["payload_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                for client_id, item in payload["clients"].items():
                    client_train_payload[int(client_id)] = item
            if set(client_train_payload) != set(
                range(1, CONFIG["num_clients"] + 1)
            ):
                raise RuntimeError("Incomplete client train results")
            local_proxy_states = [
                client_train_payload[client_id]["proxy_state"]
                for client_id in range(1, CONFIG["num_clients"] + 1)
            ]
            weights = [
                DATA["clients"][str(client_id)]["train_examples"]
                for client_id in range(1, CONFIG["num_clients"] + 1)
            ]
            global_proxy_state = aggregate_proxy_states(
                local_proxy_states,
                weights,
            )
            for client_id in personalized_states:
                personalized_states[client_id] = client_train_payload[client_id][
                    "personalized_state"
                ]
            aggregated_proxy_path = (
                CACHE_DIR / f"round_{round_index}_aggregated_proxy.pt"
            )
            torch.save(global_proxy_state, aggregated_proxy_path)
            validation_task_ids = pool.submit_both(
                "validate_proxy",
                global_proxy_path=str(aggregated_proxy_path),
                round=round_index,
            )
            validation_results = pool.collect(validation_task_ids)
            global_validation_by_client = {}
            for worker_result in validation_results.values():
                interprocess_payload_bytes += worker_result["payload_bytes"]
                payload = torch.load(
                    worker_result["payload_path"],
                    map_location="cpu",
                    weights_only=False,
                )
                for client_id, evaluation in payload.items():
                    global_validation_by_client[int(client_id)] = evaluation
            if set(global_validation_by_client) != set(
                range(1, CONFIG["num_clients"] + 1)
            ):
                raise RuntimeError("Incomplete global validation results")
            global_validation = combine_evaluations(
                [
                    global_validation_by_client[client_id]
                    for client_id in range(1, CONFIG["num_clients"] + 1)
                ]
            )

            round_loss_weight = 0
            loss_sums = {
                "personalized_label_loss": 0.0,
                "proxy_label_loss": 0.0,
                "adaptive_distillation_loss": 0.0,
                "personalized_total_loss": 0.0,
                "proxy_total_loss": 0.0,
            }
            for client_id in range(1, CONFIG["num_clients"] + 1):
                result = client_train_payload[client_id]["result"]
                client_data = DATA["clients"][str(client_id)]
                train_examples = client_data["train_examples"]
                round_loss_weight += train_examples
                for key in loss_sums:
                    loss_sums[key] += result[key] * train_examples
                total_local_train_seconds += result["train_seconds"]
                total_personal_validation_seconds += result["validation_seconds"]
                total_global_validation_seconds += global_validation_by_client[
                    client_id
                ]["seconds"]
                gpu_id = next(
                    gpu
                    for gpu, clients in client_assignment.items()
                    if client_id in clients
                )
                concurrent_streams = initialization_results[gpu_id][
                    "selected_stream_count"
                ]
                cache_mode = initialization_results[gpu_id]["cache"]["mode"]
                history_local_epoch.append(
                    {
                        "phase": "federated_mutual_learning",
                        "round": round_index,
                        "client_id": client_id,
                        "architecture": CONFIG["client_architectures"][
                            client_id - 1
                        ],
                        "gpu_id": gpu_id,
                        "concurrent_streams": concurrent_streams,
                        "gpu_cache_mode": cache_mode,
                        "local_epoch": 1,
                        "train_examples": train_examples,
                        "processed_examples_with_padding": result[
                            "processed_examples_with_padding"
                        ],
                        "optimizer_steps": result["optimizer_steps"],
                        "sampler_padding_rows": 0,
                        "hard_loss": result["personalized_label_loss"],
                        "soft_loss": result["adaptive_distillation_loss"],
                        "proximal_loss": 0.0,
                        "total_loss": result["personalized_total_loss"],
                        "personalized_label_loss": result[
                            "personalized_label_loss"
                        ],
                        "proxy_label_loss": result["proxy_label_loss"],
                        "adaptive_distillation_loss": result[
                            "adaptive_distillation_loss"
                        ],
                        "personalized_total_loss": result[
                            "personalized_total_loss"
                        ],
                        "proxy_total_loss": result["proxy_total_loss"],
                        "epoch_seconds": result["train_seconds"],
                    }
                )
                local_evaluation = result["personalized_validation"]
                global_evaluation = global_validation_by_client[client_id]
                history_client.append(
                    {
                        "round": round_index,
                        "client_id": client_id,
                        "architecture": CONFIG["client_architectures"][
                            client_id - 1
                        ],
                        "gpu_id": gpu_id,
                        "concurrent_streams": concurrent_streams,
                        "gpu_cache_mode": cache_mode,
                        "train_examples": train_examples,
                        "validation_examples": client_data[
                            "validation_examples"
                        ],
                        "hard_loss": result["personalized_label_loss"],
                        "soft_loss": result["adaptive_distillation_loss"],
                        "proximal_loss": 0.0,
                        "total_loss": result["personalized_total_loss"],
                        "proxy_label_loss": result["proxy_label_loss"],
                        "proxy_total_loss": result["proxy_total_loss"],
                        "local_train_seconds": result["train_seconds"],
                        "local_validation_seconds": result[
                            "validation_seconds"
                        ],
                        "global_validation_seconds": global_evaluation[
                            "seconds"
                        ],
                        **prefixed_metrics(
                            "local_validation",
                            local_evaluation,
                        ),
                        **prefixed_metrics(
                            "global_validation",
                            global_evaluation,
                        ),
                    }
                )

            averaged_losses = {
                key: value / max(round_loss_weight, 1)
                for key, value in loss_sums.items()
            }
            local_rows = [
                row for row in history_client if row["round"] == round_index
            ]
            local_accuracies = [
                row["local_validation_accuracy"] for row in local_rows
            ]
            local_macro_f1 = [
                row["local_validation_macro_f1"] for row in local_rows
            ]
            memory_fields = {}
            for gpu_id, worker_result in train_results.items():
                for key, value in worker_result["memory"].items():
                    memory_fields[f"gpu_{gpu_id}_{key}"] = int(value)
                    byte_name = key.removesuffix("_bytes")
                    memory_fields[f"gpu_{gpu_id}_{byte_name}_mib"] = value / MIB
            allocated_peaks = [
                train_results[gpu_id]["memory"]["peak_allocated_bytes"]
                for gpu_id in range(2)
            ]
            reserved_peaks = [
                train_results[gpu_id]["memory"]["peak_reserved_bytes"]
                for gpu_id in range(2)
            ]
            round_seconds = time.perf_counter() - round_started
            cumulative_training_seconds += round_seconds
            cumulative_communication_bytes += fl_bytes_per_round
            round_row = {
                "round": round_index,
                "hard_loss": averaged_losses["personalized_label_loss"],
                "soft_loss": averaged_losses["adaptive_distillation_loss"],
                "proximal_loss": 0.0,
                "total_loss": averaged_losses["personalized_total_loss"],
                **averaged_losses,
                "best_local_client_accuracy": max(local_accuracies),
                "worst_local_client_accuracy": min(local_accuracies),
                "mean_local_client_accuracy": float(np.mean(local_accuracies)),
                "best_local_client_macro_f1": max(local_macro_f1),
                "worst_local_client_macro_f1": min(local_macro_f1),
                "mean_local_client_macro_f1": float(np.mean(local_macro_f1)),
                "global_validation_loss": global_validation["loss"],
                **{
                    f"global_validation_{key}": value
                    for key, value in global_validation["metrics"].items()
                },
                "round_seconds": round_seconds,
                "cumulative_training_seconds": cumulative_training_seconds,
                "communication_bytes_round": fl_bytes_per_round,
                "communication_mib_round": fl_bytes_per_round / MIB,
                "communication_bytes_cumulative": cumulative_communication_bytes,
                "communication_mib_cumulative": cumulative_communication_bytes / MIB,
                "gpu_0_worker_wall_seconds": train_results[0][
                    "worker_wall_seconds"
                ],
                "gpu_1_worker_wall_seconds": train_results[1][
                    "worker_wall_seconds"
                ],
                "gpu_0_idle_seconds": max(
                    0.0,
                    round_seconds - train_results[0]["worker_wall_seconds"],
                ),
                "gpu_1_idle_seconds": max(
                    0.0,
                    round_seconds - train_results[1]["worker_wall_seconds"],
                ),
                "peak_allocated_max_gpu_bytes": int(max(allocated_peaks)),
                "peak_allocated_max_gpu_mib": max(allocated_peaks) / MIB,
                "peak_allocated_sum_gpus_bytes": int(sum(allocated_peaks)),
                "peak_allocated_sum_gpus_mib": sum(allocated_peaks) / MIB,
                "peak_reserved_max_gpu_bytes": int(max(reserved_peaks)),
                "peak_reserved_max_gpu_mib": max(reserved_peaks) / MIB,
                "peak_reserved_sum_gpus_bytes": int(sum(reserved_peaks)),
                "peak_reserved_sum_gpus_mib": sum(reserved_peaks) / MIB,
                **memory_fields,
            }
            history_round.append(round_row)
            checkpoint = {
                "model_state_dict": clone_state(global_proxy_state),
                "personalized_model_state_dicts": {
                    str(client_id): clone_state(state)
                    for client_id, state in personalized_states.items()
                },
                "initial_model_state_dicts_by_family": {
                    family: clone_state(state)
                    for family, state in initial_states.items()
                },
                "initialization_hashes": initialization_hashes,
                "round": round_index,
                "config": CONFIG,
                "feature_columns": FEATURE_COLUMNS,
                "label_mapping": LABEL_MAPPING,
                "validation_metrics": {
                    "loss": global_validation["loss"],
                    **global_validation["metrics"],
                },
                "model_metadata": model_metadata_by_family,
                "client_assignment": {
                    str(gpu_id): list(clients)
                    for gpu_id, clients in client_assignment.items()
                },
            }
            torch.save(checkpoint, OUT / "checkpoints" / "last.pt")
            current_metric = global_validation["metrics"]["macro_f1"]
            if current_metric > best_metric:
                best_metric = current_metric
                best_round = round_index
                torch.save(checkpoint, OUT / "checkpoints" / "best.pt")
            logger.info(
                "Round %02d/%02d | proxy val macro-F1=%.6f | seconds=%.2f",
                round_index,
                CONFIG["communication_rounds"],
                current_metric,
                round_seconds,
            )

        training_and_validation_seconds = time.perf_counter() - training_started
        fine_tune_task_ids = pool.submit_both(
            "finetune",
            checkpoint_path=str(OUT / "checkpoints" / "best.pt"),
        )
        fine_tune_worker_results = pool.collect(fine_tune_task_ids)
        fine_tuned_states = {}
        for worker_result in fine_tune_worker_results.values():
            interprocess_payload_bytes += worker_result["payload_bytes"]
            payload = torch.load(
                worker_result["payload_path"],
                map_location="cpu",
                weights_only=False,
            )
            for client_id, item in payload["clients"].items():
                client_id_int = int(client_id)
                fine_tuned_states[client_id_int] = item["personalized_state"]
                result = item["result"]
                fine_tune_results[client_id] = result
                gpu_id = next(
                    gpu
                    for gpu, clients in client_assignment.items()
                    if client_id_int in clients
                )
                history_local_epoch.append(
                    {
                        "phase": "final_personalized_finetune",
                        "round": CONFIG["communication_rounds"] + 1,
                        "client_id": client_id_int,
                        "architecture": CONFIG["client_architectures"][
                            client_id_int - 1
                        ],
                        "gpu_id": gpu_id,
                        "concurrent_streams": initialization_results[gpu_id][
                            "selected_stream_count"
                        ],
                        "gpu_cache_mode": initialization_results[gpu_id][
                            "cache"
                        ]["mode"],
                        "local_epoch": 1,
                        "train_examples": DATA["clients"][client_id][
                            "train_examples"
                        ],
                        "processed_examples_with_padding": result[
                            "processed_examples_with_padding"
                        ],
                        "optimizer_steps": result["optimizer_steps"],
                        "sampler_padding_rows": 0,
                        "hard_loss": result["personalized_label_loss"],
                        "soft_loss": 0.0,
                        "proximal_loss": 0.0,
                        "total_loss": result["personalized_total_loss"],
                        "personalized_label_loss": result[
                            "personalized_label_loss"
                        ],
                        "proxy_label_loss": 0.0,
                        "adaptive_distillation_loss": 0.0,
                        "personalized_total_loss": result[
                            "personalized_total_loss"
                        ],
                        "proxy_total_loss": 0.0,
                        "epoch_seconds": result["train_seconds"],
                    }
                )
        if set(fine_tuned_states) != set(
            range(1, CONFIG["num_clients"] + 1)
        ):
            raise RuntimeError("Incomplete fine-tuned personalized states")
        best_checkpoint = torch.load(
            OUT / "checkpoints" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        best_checkpoint["personalized_model_state_dicts_before_finetune"] = (
            best_checkpoint["personalized_model_state_dicts"]
        )
        best_checkpoint["personalized_model_state_dicts"] = {
            str(client_id): clone_state(state)
            for client_id, state in fine_tuned_states.items()
        }
        best_checkpoint["fine_tune_epochs"] = CONFIG[
            "final_personalized_finetune_epochs"
        ]
        torch.save(best_checkpoint, OUT / "checkpoints" / "best.pt")

        final_test_started = time.perf_counter()
        evaluation_plan = exact_evaluation_assignment(benchmark_by_gpu)
        test_examples = DATA["test"]["examples"]
        midpoint = test_examples // 2
        final_task_ids = pool.submit_both(
            "final_evaluation",
            payload_by_gpu={
                0: {
                    "evaluation_clients": list(
                        evaluation_plan["assignment"][0]
                    ),
                    "proxy_test_shard": [0, midpoint],
                },
                1: {
                    "evaluation_clients": list(
                        evaluation_plan["assignment"][1]
                    ),
                    "proxy_test_shard": [midpoint, test_examples],
                },
            },
            checkpoint_path=str(OUT / "checkpoints" / "best.pt"),
        )
        final_worker_results = pool.collect(final_task_ids)
        final_payloads = {}
        for gpu_id, worker_result in final_worker_results.items():
            interprocess_payload_bytes += worker_result["payload_bytes"]
            final_payloads[gpu_id] = torch.load(
                worker_result["payload_path"],
                map_location="cpu",
                weights_only=False,
            )
        proxy_test = combine_evaluations(
            [final_payloads[gpu_id]["proxy_shard"] for gpu_id in range(2)]
        )
        personalized_test_metrics = {}
        for payload in final_payloads.values():
            personalized_test_metrics.update(payload["personalized"])
        if set(map(int, personalized_test_metrics)) != set(
            range(1, CONFIG["num_clients"] + 1)
        ):
            raise RuntimeError("Incomplete personalized final-test results")
        final_test_seconds = time.perf_counter() - final_test_started

        save_history("history_local_epoch", history_local_epoch)
        save_history("history_client", history_client)
        save_history("history_round", history_round)
        report_json, report_rows = classification_report_from_confusion(
            proxy_test["confusion_matrix"]
        )
        write_json(OUT / "metrics" / "classification_report.json", report_json)
        pd.DataFrame(report_rows).to_csv(
            OUT / "metrics" / "classification_report.csv",
            index=False,
        )
        class_names = [
            LABEL_MAPPING[class_id] for class_id in range(CONFIG["num_classes"])
        ]
        pd.DataFrame(
            proxy_test["confusion_matrix"],
            index=class_names,
            columns=class_names,
        ).to_csv(
            OUT / "metrics" / "confusion_matrix.csv",
            index_label="true_label",
        )
        np.save(
            OUT / "metrics" / "confusion_matrix.npy",
            proxy_test["confusion_matrix"].astype(np.int64),
        )

        communication = {
            "logical_federated": {
                "proxy_model_state_bytes": proxy_state_bytes,
                "clients_per_round": CONFIG["num_clients"],
                "rounds": CONFIG["communication_rounds"],
                "bytes_per_round": fl_bytes_per_round,
                "mib_per_round": fl_bytes_per_round / MIB,
                "total_bytes": cumulative_communication_bytes,
                "total_mib": cumulative_communication_bytes / MIB,
                "formula": "2 * clients * proxy_model_state_bytes",
                "excludes_personalized_models": True,
            },
            "ddp_gradient_allreduce_estimate": {
                "enabled": False,
                "total_bytes": 0,
                "total_mib": 0.0,
                "reason": "client_parallel execution has no gradient collective",
            },
            "client_parallel_interprocess_payload": {
                "total_bytes": interprocess_payload_bytes,
                "total_mib": interprocess_payload_bytes / MIB,
                "excluded_from_logical_federated_cost": True,
            },
        }
        write_json(OUT / "metrics" / "communication_costs.json", communication)
        pd.DataFrame(
            [
                {
                    "round": row["round"],
                    "logical_fl_bytes_round": row["communication_bytes_round"],
                    "logical_fl_mib_round": row["communication_mib_round"],
                    "logical_fl_bytes_cumulative": row[
                        "communication_bytes_cumulative"
                    ],
                    "logical_fl_mib_cumulative": row[
                        "communication_mib_cumulative"
                    ],
                }
                for row in history_round
            ]
        ).to_csv(OUT / "metrics" / "communication_costs.csv", index=False)

        plotting_started = time.perf_counter()
        create_plots(
            history_round,
            proxy_test["confusion_matrix"],
            report_rows,
        )
        plotting_seconds = time.perf_counter() - plotting_started
        monitor.stop()
        utilization_summary = monitor.summary()
        personalized_f1 = {
            client_id: values["macro_f1"]
            for client_id, values in personalized_test_metrics.items()
        }
        runtime = {
            "preprocessing_seconds": MANIFEST["preprocessing_seconds"],
            "training_and_validation_seconds": training_and_validation_seconds,
            "final_test_seconds": final_test_seconds,
            "plotting_seconds": plotting_seconds,
            "client_parallel_runtime_seconds": time.perf_counter()
            - runtime_started,
            "subprocess_wall_seconds": None,
            "total_notebook_pipeline_seconds": time.perf_counter()
            - MANIFEST["pipeline_started_perf_counter"],
            "total_local_training_seconds": total_local_train_seconds,
            "total_personalized_validation_seconds": (
                total_personal_validation_seconds
            ),
            "total_global_on_client_validation_seconds": (
                total_global_validation_seconds
            ),
            "workload_benchmark": worker_benchmarks,
            "client_assignment": {
                str(gpu_id): list(clients)
                for gpu_id, clients in client_assignment.items()
            },
            "work_order": {
                str(gpu_id): list(clients)
                for gpu_id, clients in work_order.items()
            },
            "predicted_worker_load_seconds": assignment_plan[
                "predicted_load_seconds"
            ],
            "actual_worker_wall_seconds_by_round": round_worker_wall,
            "initialization_by_worker": initialization_results,
            "gpu_utilization": utilization_summary,
            "final_evaluation_assignment": {
                str(gpu_id): list(clients)
                for gpu_id, clients in evaluation_plan["assignment"].items()
            },
            "final_test_cache": {
                str(gpu_id): {
                    "seconds": final_payloads[gpu_id]["test_cache_seconds"],
                    "bytes": final_payloads[gpu_id]["test_cache_bytes"],
                }
                for gpu_id in range(2)
            },
            "round_seconds": {
                str(row["round"]): row["round_seconds"] for row in history_round
            },
            "client_seconds": {
                f"round_{row['round']}_client_{row['client_id']}": {
                    "gpu_id": row["gpu_id"],
                    "local_train_seconds": row["local_train_seconds"],
                    "local_validation_seconds": row[
                        "local_validation_seconds"
                    ],
                    "global_validation_seconds": row[
                        "global_validation_seconds"
                    ],
                }
                for row in history_client
            },
        }
        write_json(OUT / "metrics" / "runtime_breakdown.json", runtime)
        required_outputs = [
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
        ]
        summary = {
            "run_name": CONFIG["run_name"],
            "scenario_name": CONFIG["scenario_name"],
            "model_family": CONFIG["scenario_model_family"],
            "proxy_model_family": "cnn1d",
            "execution_mode": "client_parallel",
            "status": "completed",
            "config": CONFIG,
            "environment": environment_by_worker,
            "model_metadata": model_metadata_by_family,
            "initialization_hashes": initialization_hashes,
            "initialization_contract": {
                "seed": CONFIG["initialization_seed"],
                "same_family_same_initial_state": True,
                "initial_states_embedded_in_checkpoints": True,
            },
            "client_assignment": runtime["client_assignment"],
            "gpu_cache_and_streams": initialization_results,
            "dataset": {
                "global_train_rows": DATA["global_train_rows"],
                "train_rows_after_validation": sum(
                    item["train_examples"] for item in DATA["clients"].values()
                ),
                "validation_rows": sum(
                    item["validation_examples"]
                    for item in DATA["clients"].values()
                ),
                "test_rows": DATA["test"]["examples"],
                "test_class_counts": DATA["test"]["class_counts"],
            },
            "best_round": best_round,
            "best_validation_macro_f1": best_metric,
            "final_test_metrics": {
                "loss": proxy_test["loss"],
                **proxy_test["metrics"],
            },
            "personalized_test_metrics": personalized_test_metrics,
            "personalized_test_macro_f1_summary": {
                "mean": float(np.mean(list(personalized_f1.values()))),
                "best_client_id": max(personalized_f1, key=personalized_f1.get),
                "best": max(personalized_f1.values()),
                "worst_client_id": min(personalized_f1, key=personalized_f1.get),
                "worst": min(personalized_f1.values()),
            },
            "fine_tune": fine_tune_results,
            "runtime_breakdown": runtime,
            "communication_estimate": communication,
            "output_paths": required_outputs,
        }
        write_json(OUT / "metrics" / "summary.json", summary)
        logger.info(
            "Final proxy test accuracy=%.6f macro-F1=%.6f",
            proxy_test["metrics"]["accuracy"],
            proxy_test["metrics"]["macro_f1"],
        )
    except BaseException:
        run_failed = True
        raise
    finally:
        if monitor.thread.is_alive():
            monitor.stop()
        if run_failed:
            try:
                pool.shutdown()
            except Exception:
                logger.error(
                    "Worker shutdown also failed after the primary error:\n%s",
                    traceback.format_exc(),
                )
            try:
                merge_worker_logs()
            except Exception:
                logger.error(
                    "Worker log merge also failed after the primary error:\n%s",
                    traceback.format_exc(),
                )
        else:
            pool.shutdown()
            merge_worker_logs()


def main():
    logger = setup_logger(
        OUT / "logs" / "run.log",
        "client_parallel_coordinator",
        include_stdout=True,
    )
    try:
        logger.info("Client-parallel coordinator started")
        coordinator_run(logger)
    except Exception:
        logger.error("Client-parallel run failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
