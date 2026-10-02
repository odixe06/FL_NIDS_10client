#!/usr/bin/env python3
"""Two-process DDP runtime embedded by the generated Kaggle notebooks."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


MANIFEST_PATH = Path(os.environ["FD_IDS_MANIFEST"])
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
CONFIG = MANIFEST["config"]
CLIENT_METADATA = MANIFEST["client_metadata"]
TEST_METADATA = MANIFEST["test_metadata"]
DATASET_SUMMARY = MANIFEST["dataset_summary"]
LABEL_MAPPING = {int(key): value for key, value in MANIFEST["label_mapping"].items()}
CLASS_NAMES = [LABEL_MAPPING[class_id] for class_id in range(CONFIG["num_classes"])]
BENIGN_ID = int(MANIFEST["benign_id"])
FEATURE_COLUMNS = CONFIG["feature_columns"]
OUT = Path(MANIFEST["out_dir"])


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == CONFIG["world_size"] == 2
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=CONFIG["distributed_backend"], init_method="env://")
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(CONFIG["omp_num_threads_per_process"])
    return rank, local_rank, world_size, device


def setup_logger(rank):
    log_path = (
        OUT / "logs" / "run.log"
        if rank == 0
        else OUT / "logs" / f"rank_{rank}.log"
    )
    handlers = [
        logging.FileHandler(
            log_path,
            mode="a" if rank == 0 else "w",
            encoding="utf-8",
        ),
    ]
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("fd_ids_ddp")


class IndexedMemmapDataset(Dataset):
    """Disk-backed dataset that safely reopens arrays in DataLoader workers."""

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


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding or duplicating samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


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


def make_train_loader(dataset, rank, world_size, seed):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size_per_gpu"],
        sampler=sampler,
        shuffle=False,
        num_workers=CONFIG["num_workers_per_process"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=False,
    )
    return loader, sampler


def make_eval_loader(dataset, rank, world_size, seed):
    sampler = DistributedEvalSampler(dataset, rank, world_size)
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size_per_gpu"],
        sampler=sampler,
        shuffle=False,
        num_workers=CONFIG["num_workers_per_process"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=False,
    )


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

    attack_ids = [
        class_id
        for class_id in range(CONFIG["num_classes"])
        if class_id != BENIGN_ID
    ]
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
            safe_divide(
                binary_tp + binary_tn,
                binary_tp + binary_tn + binary_fp + binary_fn,
            )
        ),
        "binary_attack_precision": binary_precision,
        "binary_attack_recall": binary_recall,
        "binary_attack_f1": binary_f1,
        "binary_attack_fpr": float(safe_divide(binary_fp, binary_fp + binary_tn)),
        "binary_attack_fnr": float(safe_divide(binary_fn, binary_fn + binary_tp)),
    }


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


class TransformerClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        d_model = 64
        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_features, d_model)
        )
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
    if CONFIG["model_family"] == "gru":
        return GRUClassifier(len(FEATURE_COLUMNS), CONFIG["num_classes"])
    if CONFIG["model_family"] == "transformer":
        return TransformerClassifier(len(FEATURE_COLUMNS), CONFIG["num_classes"])
    if CONFIG["model_family"] == "cnn1d":
        return CNN1DClassifier(len(FEATURE_COLUMNS), CONFIG["num_classes"])
    raise ValueError(f"Unknown model family: {CONFIG['model_family']}")


def broadcast_model_state(model, source_rank=0):
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=source_rank)


def new_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def proximal_loss(student, teacher, device):
    squared_distance = torch.zeros((), device=device)
    for student_parameter, teacher_parameter in zip(
        student.parameters(), teacher.parameters()
    ):
        squared_distance = squared_distance + torch.sum(
            (student_parameter - teacher_parameter.detach()) ** 2
        )
    return 0.5 * CONFIG["mu"] * squared_distance


def reduce_max_seconds(seconds, device):
    value = torch.tensor(seconds, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


@torch.no_grad()
def evaluate_distributed(model, dataset, rank, world_size, device, seed):
    model.eval()
    loader = make_eval_loader(dataset, rank, world_size, seed)
    matrix = torch.zeros(
        (CONFIG["num_classes"], CONFIG["num_classes"]),
        dtype=torch.int64,
        device=device,
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    examples = torch.zeros((), dtype=torch.int64, device=device)

    dist.barrier()
    started = time.perf_counter()
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=1)
        batch_examples = targets.size(0)
        loss_sum += loss.to(torch.float64) * batch_examples
        examples += batch_examples
        encoded = targets.to(torch.int64) * CONFIG["num_classes"] + predictions
        matrix += torch.bincount(
            encoded,
            minlength=CONFIG["num_classes"] ** 2,
        ).reshape(CONFIG["num_classes"], CONFIG["num_classes"])
    dist.barrier()
    elapsed = reduce_max_seconds(time.perf_counter() - started, device)
    dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(examples, op=dist.ReduceOp.SUM)

    matrix_numpy = matrix.cpu().numpy()
    example_count = int(examples.item())
    assert example_count == len(dataset)
    return {
        "loss": float(loss_sum.item() / max(example_count, 1)),
        "examples": example_count,
        "metrics": metrics_from_confusion(matrix_numpy),
        "confusion_matrix": matrix_numpy,
        "seconds": elapsed,
    }


def train_one_client(
    global_teacher,
    metadata,
    round_number,
    rank,
    local_rank,
    world_size,
    device,
    logger,
):
    client_id = metadata["client_id"]
    dataset = dataset_from_metadata(metadata, "train")
    loader, sampler = make_train_loader(
        dataset,
        rank,
        world_size,
        seed=CONFIG["seed"] + round_number * 10_000 + client_id,
    )
    student_core = build_model().to(device)
    if CONFIG["model_family"] == "cnn1d":
        # Match BatchNorm statistics to the effective global DDP batch.
        student_core = nn.SyncBatchNorm.convert_sync_batchnorm(student_core)
    student_core.load_state_dict(global_teacher.state_dict())
    student = DDP(
        student_core,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=CONFIG["learning_rate"],
    )
    scaler = new_grad_scaler()
    global_teacher.eval()
    local_epoch_rows = []
    overall_sums = torch.zeros(4, dtype=torch.float64, device=device)
    overall_examples = torch.zeros((), dtype=torch.int64, device=device)

    dist.barrier()
    client_started = time.perf_counter()
    for local_epoch in range(1, CONFIG["local_epochs"] + 1):
        sampler.set_epoch(round_number * 100 + local_epoch)
        student.train()
        epoch_sums = torch.zeros(4, dtype=torch.float64, device=device)
        epoch_examples = torch.zeros((), dtype=torch.int64, device=device)
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
                            student_logits / CONFIG["temperature"],
                            dim=1,
                        ),
                        F.softmax(
                            teacher_logits / CONFIG["temperature"],
                            dim=1,
                        ),
                        reduction="batchmean",
                    )
                    * CONFIG["temperature"] ** 2
                )
            prox_loss = proximal_loss(student.module, global_teacher, device)
            total_loss = (
                CONFIG["lambda_hard"] * hard_loss
                + (1.0 - CONFIG["lambda_hard"]) * soft_loss
                + CONFIG["beta"] * prox_loss
            )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_examples = targets.size(0)
            values = torch.stack(
                [
                    hard_loss.detach().to(torch.float64),
                    soft_loss.detach().to(torch.float64),
                    prox_loss.detach().to(torch.float64),
                    total_loss.detach().to(torch.float64),
                ]
            )
            epoch_sums += values * batch_examples
            overall_sums += values * batch_examples
            epoch_examples += batch_examples
            overall_examples += batch_examples

        epoch_seconds = reduce_max_seconds(
            time.perf_counter() - epoch_started,
            device,
        )
        dist.all_reduce(epoch_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_examples, op=dist.ReduceOp.SUM)
        reduced_examples = int(epoch_examples.item())
        reduced_sums = epoch_sums.cpu().numpy()
        epoch_row = {
            "round": round_number,
            "client_id": client_id,
            "local_epoch": local_epoch,
            "train_examples": reduced_examples,
            "optimizer_steps": len(loader),
            "sampler_padding_rows": int(sampler.total_size - len(dataset)),
            "hard_loss": float(reduced_sums[0] / max(reduced_examples, 1)),
            "soft_loss": float(reduced_sums[1] / max(reduced_examples, 1)),
            "proximal_loss": float(reduced_sums[2] / max(reduced_examples, 1)),
            "total_loss": float(reduced_sums[3] / max(reduced_examples, 1)),
            "epoch_seconds": epoch_seconds,
        }
        local_epoch_rows.append(epoch_row)
        if rank == 0:
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

    dist.all_reduce(overall_sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(overall_examples, op=dist.ReduceOp.SUM)
    broadcast_model_state(student.module, source_rank=0)
    train_seconds = reduce_max_seconds(time.perf_counter() - client_started, device)

    local_validation_dataset = dataset_from_metadata(metadata, "validation")
    local_validation_result = evaluate_distributed(
        student.module,
        local_validation_dataset,
        rank,
        world_size,
        device,
        seed=CONFIG["seed"] + round_number * 10_000 + client_id + 1_000_000,
    )
    local_state = None
    if rank == 0:
        local_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in student.module.state_dict().items()
        }
    reduced_sums = overall_sums.cpu().numpy()
    reduced_examples = int(overall_examples.item())
    client_stats = {
        "round": round_number,
        "client_id": client_id,
        "train_rows": metadata["train_rows"],
        "validation_rows": metadata["val_rows"],
        "train_examples_processed": reduced_examples,
        "sampler_padding_rows_per_epoch": int(sampler.total_size - len(dataset)),
        "hard_loss": float(reduced_sums[0] / max(reduced_examples, 1)),
        "soft_loss": float(reduced_sums[1] / max(reduced_examples, 1)),
        "proximal_loss": float(reduced_sums[2] / max(reduced_examples, 1)),
        "total_loss": float(reduced_sums[3] / max(reduced_examples, 1)),
        "local_train_seconds": train_seconds,
        "local_validation_loss": local_validation_result["loss"],
        "local_validation_seconds": local_validation_result["seconds"],
    }
    for metric_name, metric_value in local_validation_result["metrics"].items():
        client_stats[f"local_validation_{metric_name}"] = metric_value
    if rank == 0:
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
        student_core,
        optimizer,
        scaler,
        loader,
        sampler,
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
    return {
        name: aggregate[name].to(dtype=reference_tensor.dtype)
        for name, reference_tensor in reference_state.items()
    }


def persist_histories(round_history, client_history, local_epoch_history):
    pd.DataFrame(round_history).to_csv(
        OUT / "metrics" / "history_round.csv",
        index=False,
    )
    pd.DataFrame(client_history).to_csv(
        OUT / "metrics" / "history_client.csv",
        index=False,
    )
    pd.DataFrame(local_epoch_history).to_csv(
        OUT / "metrics" / "history_local_epoch.csv",
        index=False,
    )
    write_json(OUT / "metrics" / "history_round.json", round_history)
    write_json(OUT / "metrics" / "history_client.json", client_history)
    write_json(OUT / "metrics" / "history_local_epoch.json", local_epoch_history)


def render_plots(
    round_history,
    distribution_frame,
    test_matrix,
    classification_report_frame,
):
    round_frame = pd.DataFrame(round_history)
    class_count_matrix = (
        distribution_frame.pivot(
            index="client_id",
            columns="class_name",
            values="total_count",
        )
        .reindex(
            index=range(1, CONFIG["num_clients"] + 1),
            columns=CLASS_NAMES,
        )
        .fillna(0)
        .to_numpy()
    )
    plt.figure(figsize=(18, 5.5))
    plt.imshow(np.log10(class_count_matrix + 1), aspect="auto", cmap="magma")
    plt.colorbar(label="log10(count + 1)")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90, fontsize=7)
    plt.yticks(
        range(CONFIG["num_clients"]),
        [f"Client {index}" for index in range(1, CONFIG["num_clients"] + 1)],
    )
    plt.xlabel("Class")
    plt.ylabel("Client")
    plt.title("Client class distribution (all pre-split rows)")
    plt.tight_layout()
    plt.savefig(OUT / "artifacts" / "class_distribution.png", dpi=160)
    plt.show()
    plt.close()

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
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(round_frame["round"], round_frame["train_hard_loss"], label="Hard loss")
    plt.plot(
        round_frame["round"],
        round_frame["train_soft_loss"],
        label="Soft KD loss",
    )
    plt.plot(
        round_frame["round"],
        round_frame["train_proximal_loss"],
        label="FedProx loss",
    )
    plt.plot(
        round_frame["round"],
        round_frame["train_total_loss"],
        label="Total loss",
    )
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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        round_frame["round"],
        round_frame["communication_mib_cumulative"],
        marker="o",
        color="darkgreen",
    )
    plt.xlabel("Communication round")
    plt.ylabel("Estimated cumulative FL communication (MiB)")
    plt.title("Raw model-state FL communication estimate")
    plt.xticks(round_frame["round"])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "artifacts" / "communication_cumulative.png", dpi=160)
    plt.show()
    plt.close()


def run(rank, local_rank, world_size, device, logger):
    run_started = time.perf_counter()
    random.seed(CONFIG["seed"] + rank)
    np.random.seed(CONFIG["seed"] + rank)
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True

    global_model = build_model().to(device)
    broadcast_model_state(global_model, source_rank=0)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in global_model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in global_model.parameters())
    model_state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_model.state_dict().values()
    )
    trainable_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in global_model.parameters()
        if parameter.requires_grad
    )
    assert trainable_parameters == CONFIG["expected_trainable_parameters"]
    model_metadata = {
        "model_family": CONFIG["model_family"],
        "model_hparams": CONFIG["model_hparams"],
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "model_state_bytes": int(model_state_bytes),
        "model_state_mib": float(model_state_bytes / 1024**2),
        "trainable_parameter_bytes": int(trainable_parameter_bytes),
    }
    if rank == 0:
        logger.info("DDP model metadata: %s", json.dumps(model_metadata, sort_keys=True))

    round_history = []
    client_history = []
    local_epoch_history = []
    best_validation_macro_f1 = -math.inf
    best_round = None
    best_validation_metrics = None
    total_train_rows = sum(item["train_rows"] for item in CLIENT_METADATA)
    communication_bytes_per_round = (
        2 * CONFIG["num_clients"] * model_state_bytes
    )
    cumulative_communication_bytes = 0
    training_started = time.perf_counter()

    for round_number in range(1, CONFIG["communication_rounds"] + 1):
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        round_started = time.perf_counter()
        aggregate = {}
        pending_client_rows = []

        for client_position, metadata in enumerate(CLIENT_METADATA):
            local_state, client_stats, epoch_rows = train_one_client(
                global_model,
                metadata,
                round_number,
                rank,
                local_rank,
                world_size,
                device,
                logger,
            )
            client_weight = metadata["train_rows"] / total_train_rows
            if rank == 0:
                add_weighted_state(
                    aggregate,
                    local_state,
                    client_weight,
                    first_client=(client_position == 0),
                )
            pending_client_rows.append(client_stats)
            if rank == 0:
                local_epoch_history.extend(epoch_rows)
            del local_state

        if rank == 0:
            aggregated_state = finalize_aggregate(
                aggregate,
                global_model.state_dict(),
            )
            global_model.load_state_dict(aggregated_state, strict=True)
            del aggregated_state
        broadcast_model_state(global_model, source_rank=0)
        del aggregate
        gc.collect()

        global_validation_matrix = np.zeros(
            (CONFIG["num_classes"], CONFIG["num_classes"]),
            dtype=np.int64,
        )
        global_validation_loss_sum = 0.0
        global_validation_examples = 0
        for metadata, client_row in zip(CLIENT_METADATA, pending_client_rows):
            validation_dataset = dataset_from_metadata(metadata, "validation")
            validation_result = evaluate_distributed(
                global_model,
                validation_dataset,
                rank,
                world_size,
                device,
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
            if rank == 0:
                client_history.append(client_row)
            del validation_dataset, validation_result

        validation_metrics = metrics_from_confusion(global_validation_matrix)
        validation_loss = global_validation_loss_sum / max(
            global_validation_examples,
            1,
        )
        weighted_train = {
            loss_name: sum(
                row[loss_name] * row["train_rows"]
                for row in pending_client_rows
            )
            / total_train_rows
            for loss_name in (
                "hard_loss",
                "soft_loss",
                "proximal_loss",
                "total_loss",
            )
        }
        dist.barrier()
        round_seconds = reduce_max_seconds(
            time.perf_counter() - round_started,
            device,
        )
        cumulative_training_seconds = time.perf_counter() - training_started
        cumulative_communication_bytes += communication_bytes_per_round
        local_peaks = torch.tensor(
            [
                int(torch.cuda.max_memory_allocated(device)),
                int(torch.cuda.max_memory_reserved(device)),
            ],
            dtype=torch.int64,
            device=device,
        )
        gathered_peaks = [
            torch.zeros_like(local_peaks) for _ in range(world_size)
        ]
        dist.all_gather(gathered_peaks, local_peaks)
        gpu_allocated_peak_bytes = [
            int(value[0].item()) for value in gathered_peaks
        ]
        gpu_reserved_peak_bytes = [
            int(value[1].item()) for value in gathered_peaks
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
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "round_seconds": round_seconds,
            "cumulative_training_seconds": cumulative_training_seconds,
            "communication_bytes_round": communication_bytes_per_round,
            "communication_mib_round": communication_bytes_per_round / 1024**2,
            "communication_bytes_cumulative": cumulative_communication_bytes,
            "communication_mib_cumulative": cumulative_communication_bytes / 1024**2,
            "cuda_gpu0_peak_allocated_bytes": gpu_allocated_peak_bytes[0],
            "cuda_gpu0_peak_allocated_mib": (
                gpu_allocated_peak_bytes[0] / 1024**2
            ),
            "cuda_gpu1_peak_allocated_bytes": gpu_allocated_peak_bytes[1],
            "cuda_gpu1_peak_allocated_mib": (
                gpu_allocated_peak_bytes[1] / 1024**2
            ),
            "cuda_peak_allocated_max_device_bytes": max(
                gpu_allocated_peak_bytes
            ),
            "cuda_peak_allocated_max_device_mib": (
                max(gpu_allocated_peak_bytes) / 1024**2
            ),
            "cuda_peak_allocated_sum_devices_bytes": sum(
                gpu_allocated_peak_bytes
            ),
            "cuda_peak_allocated_sum_devices_mib": (
                sum(gpu_allocated_peak_bytes) / 1024**2
            ),
            "cuda_gpu0_peak_reserved_bytes": gpu_reserved_peak_bytes[0],
            "cuda_gpu0_peak_reserved_mib": gpu_reserved_peak_bytes[0] / 1024**2,
            "cuda_gpu1_peak_reserved_bytes": gpu_reserved_peak_bytes[1],
            "cuda_gpu1_peak_reserved_mib": gpu_reserved_peak_bytes[1] / 1024**2,
            "cuda_peak_reserved_max_device_bytes": max(gpu_reserved_peak_bytes),
            "cuda_peak_reserved_max_device_mib": (
                max(gpu_reserved_peak_bytes) / 1024**2
            ),
            "cuda_peak_reserved_sum_devices_bytes": sum(
                gpu_reserved_peak_bytes
            ),
            "cuda_peak_reserved_sum_devices_mib": (
                sum(gpu_reserved_peak_bytes) / 1024**2
            ),
        }
        if rank == 0:
            round_history.append(round_row)
            checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in global_model.state_dict().items()
                },
                "round": round_number,
                "config": CONFIG,
                "feature_columns": FEATURE_COLUMNS,
                "label_mapping": LABEL_MAPPING,
                "validation_metrics": {
                    "loss": validation_loss,
                    **validation_metrics,
                },
                "model_metadata": model_metadata,
            }
            torch.save(checkpoint, OUT / "checkpoints" / "last.pt")
            if validation_metrics["macro_f1"] > best_validation_macro_f1:
                best_validation_macro_f1 = validation_metrics["macro_f1"]
                best_round = round_number
                best_validation_metrics = {
                    "loss": validation_loss,
                    **validation_metrics,
                }
                torch.save(checkpoint, OUT / "checkpoints" / "best.pt")
                logger.info(
                    "Saved best.pt at round %d with validation macro-F1 %.6f",
                    round_number,
                    best_validation_macro_f1,
                )
            persist_histories(
                round_history,
                client_history,
                local_epoch_history,
            )
            logger.info(
                "Round %02d/%02d complete | val_loss=%.6f val_acc=%.6f "
                "val_macro_f1=%.6f | %.2f s | FL comm=%.3f MiB cumulative",
                round_number,
                CONFIG["communication_rounds"],
                validation_loss,
                validation_metrics["accuracy"],
                validation_metrics["macro_f1"],
                round_seconds,
                cumulative_communication_bytes / 1024**2,
            )

    training_and_validation_seconds = reduce_max_seconds(
        time.perf_counter() - training_started,
        device,
    )
    if rank == 0:
        try:
            checkpoint = torch.load(
                OUT / "checkpoints" / "best.pt",
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                OUT / "checkpoints" / "best.pt",
                map_location=device,
            )
        global_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("Reloaded best.pt from round %d", checkpoint["round"])
    broadcast_model_state(global_model, source_rank=0)

    test_dataset = dataset_from_metadata(TEST_METADATA, "test")
    final_test_started = time.perf_counter()
    test_result = evaluate_distributed(
        global_model,
        test_dataset,
        rank,
        world_size,
        device,
        seed=CONFIG["seed"] + 999_999,
    )
    final_test_seconds = reduce_max_seconds(
        time.perf_counter() - final_test_started,
        device,
    )

    if rank == 0:
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
        pd.DataFrame(
            test_matrix,
            index=[f"true_{name}" for name in CLASS_NAMES],
            columns=[f"pred_{name}" for name in CLASS_NAMES],
        ).to_csv(OUT / "metrics" / "confusion_matrix.csv")
        classification_report, classification_report_frame = (
            classification_report_from_confusion(test_matrix)
        )
        write_json(
            OUT / "metrics" / "classification_report.json",
            classification_report,
        )
        classification_report_frame.to_csv(
            OUT / "metrics" / "classification_report.csv",
            index=False,
        )

        communication_rows = []
        for round_number in range(1, CONFIG["communication_rounds"] + 1):
            communication_rows.append(
                {
                    "round": round_number,
                    "model_state_bytes": model_state_bytes,
                    "participating_clients": CONFIG["num_clients"],
                    "download_bytes": CONFIG["num_clients"] * model_state_bytes,
                    "upload_bytes": CONFIG["num_clients"] * model_state_bytes,
                    "total_bytes_round": communication_bytes_per_round,
                    "total_mib_round": communication_bytes_per_round / 1024**2,
                    "cumulative_bytes": communication_bytes_per_round * round_number,
                    "cumulative_mib": (
                        communication_bytes_per_round * round_number / 1024**2
                    ),
                }
            )
        pd.DataFrame(communication_rows).to_csv(
            OUT / "metrics" / "communication_costs.csv",
            index=False,
        )
        total_optimizer_steps = int(
            sum(row["optimizer_steps"] for row in local_epoch_history)
        )
        allreduce_bytes_per_rank_per_step = int(
            2
            * (world_size - 1)
            / world_size
            * trainable_parameter_bytes
        )
        communication_summary = {
            "fl_estimation_scope": (
                "raw model state only; server download plus client upload; "
                "excludes protocol, serialization, retry and compression"
            ),
            "model_state_bytes": model_state_bytes,
            "model_state_mib": model_state_bytes / 1024**2,
            "participating_clients_per_round": CONFIG["num_clients"],
            "rounds": CONFIG["communication_rounds"],
            "fl_total_bytes_per_round": communication_bytes_per_round,
            "fl_total_mib_per_round": communication_bytes_per_round / 1024**2,
            "fl_total_bytes_all_rounds": (
                communication_bytes_per_round * CONFIG["communication_rounds"]
            ),
            "fl_total_mib_all_rounds": (
                communication_bytes_per_round
                * CONFIG["communication_rounds"]
                / 1024**2
            ),
            "ddp_estimation_scope": (
                "ring all-reduce gradient payload approximation; excludes "
                "bucket padding, buffer broadcast and NCCL protocol overhead"
            ),
            "ddp_world_size": world_size,
            "ddp_optimizer_steps": total_optimizer_steps,
            "ddp_gradient_allreduce_bytes_per_rank_per_step": (
                allreduce_bytes_per_rank_per_step
            ),
            "ddp_gradient_allreduce_bytes_all_ranks": (
                allreduce_bytes_per_rank_per_step
                * world_size
                * total_optimizer_steps
            ),
            "ddp_gradient_allreduce_mib_all_ranks": (
                allreduce_bytes_per_rank_per_step
                * world_size
                * total_optimizer_steps
                / 1024**2
            ),
        }
        write_json(
            OUT / "metrics" / "communication_costs.json",
            communication_summary,
        )
        runtime_breakdown = {
            "preprocessing_seconds": MANIFEST["preprocessing_seconds"],
            "training_and_validation_seconds": training_and_validation_seconds,
            "final_test_seconds": final_test_seconds,
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
        distribution_frame = pd.read_csv(
            OUT / "metrics" / "client_class_distribution.csv"
        )
        plotting_started = time.perf_counter()
        render_plots(
            round_history,
            distribution_frame,
            test_matrix,
            classification_report_frame,
        )
        plotting_seconds = time.perf_counter() - plotting_started
        runtime_breakdown["plotting_seconds"] = plotting_seconds
        runtime_breakdown["ddp_runtime_seconds"] = time.perf_counter() - run_started
        runtime_breakdown["total_notebook_pipeline_seconds"] = (
            MANIFEST["preprocessing_seconds"]
            + runtime_breakdown["ddp_runtime_seconds"]
        )
        write_json(
            OUT / "metrics" / "runtime_breakdown.json",
            runtime_breakdown,
        )
        output_files = {
            "best_checkpoint": "checkpoints/best.pt",
            "last_checkpoint": "checkpoints/last.pt",
            "run_log": "logs/run.log",
            "rank_1_log": "logs/rank_1.log",
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
        }
        summary = {
            "run_name": CONFIG["run_name"],
            "model_family": CONFIG["model_family"],
            "status": "completed",
            "config": CONFIG,
            "environment": {
                **MANIFEST["environment"],
                "distributed_backend": CONFIG["distributed_backend"],
                "world_size": world_size,
                "gpu_details": [
                    {
                        "index": gpu_index,
                        "name": torch.cuda.get_device_name(gpu_index),
                        "total_memory_bytes": int(
                            torch.cuda.get_device_properties(
                                gpu_index
                            ).total_memory
                        ),
                        "total_memory_mib": float(
                            torch.cuda.get_device_properties(
                                gpu_index
                            ).total_memory
                            / 1024**2
                        ),
                        "compute_capability": (
                            f"{torch.cuda.get_device_capability(gpu_index)[0]}."
                            f"{torch.cuda.get_device_capability(gpu_index)[1]}"
                        ),
                    }
                    for gpu_index in range(world_size)
                ],
            },
            "model_metadata": model_metadata,
            "dataset": DATASET_SUMMARY,
            "best_round": best_round,
            "best_validation_metrics": best_validation_metrics,
            "final_test_metrics": test_metrics,
            "runtime_seconds": runtime_breakdown,
            "communication": communication_summary,
            "metric_units": {
                "rates": "fraction_in_[0,1]",
                "time": "seconds",
                "communication_binary": "MiB (1 MiB = 1,048,576 bytes)",
                "communication_exact": "bytes",
            },
            "outputs": output_files,
        }
        write_json(OUT / "metrics" / "summary.json", summary)
        logger.info("Final test metrics: %s", json.dumps(test_metrics, sort_keys=True))
        logger.info(
            "DDP training completed | best round=%d macro-F1=%.6f | %.2f s",
            best_round,
            best_validation_macro_f1,
            runtime_breakdown["ddp_runtime_seconds"],
        )
        for output_path in sorted(OUT.rglob("*")):
            if output_path.is_file():
                logger.info(
                    "OUTPUT %s | %s bytes",
                    output_path.relative_to(OUT),
                    f"{output_path.stat().st_size:,}",
                )
    dist.barrier()


def main():
    rank = local_rank = None
    logger = None
    try:
        rank, local_rank, world_size, device = setup_distributed()
        logger = setup_logger(rank)
        logger.info(
            "Starting DDP rank %d/%d on %s | per-rank batch=%d",
            rank,
            world_size,
            torch.cuda.get_device_name(local_rank),
            CONFIG["batch_size_per_gpu"],
        )
        run(rank, local_rank, world_size, device, logger)
    except Exception:
        if logger is not None:
            logger.error("DDP rank failed: %s", traceback.format_exc())
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
