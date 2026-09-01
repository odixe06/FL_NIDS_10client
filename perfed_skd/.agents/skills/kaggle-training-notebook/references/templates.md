# PyTorch DDP on Kaggle 2×T4

Use these patterns only after the user approves the complete specification.
Adapt every task-specific section and remove all illustrative values from the
delivered notebook.

## Contents

- [Runtime paths](#runtime-paths)
- [Environment check](#cell-1--environment-check)
- [Configuration and parent logging](#cell-2--configuration-and-parent-logging)
- [DDP launcher](#notebook-launcher)
- [Distributed initialization](#ddp-entry-point--initialization-and-logging)
- [Samplers and loaders](#ddp-entry-point--samplers-and-loaders)
- [Model and AMP](#ddp-entry-point--model-and-amp)
- [Training](#ddp-entry-point--training)
- [Exact distributed evaluation](#ddp-entry-point--exact-distributed-evaluation)
- [Outputs and inline plots](#outputs-and-inline-plots)
- [Validation checklist](#validation-checklist)

## Runtime paths

- Read from the exact confirmed `/kaggle/input/...` path.
- Persist under `/kaggle/working/{run_name}/`.
- Write the DDP script, manifest, and large disposable arrays under
  `/kaggle/temp/`.
- Require exactly two T4 devices unless the user explicitly changes topology.

## Cell 1 — Environment check

Keep the parent kernel CPU-only after checking hardware. The training
subprocess owns the CUDA models.

```python
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
        f"VRAM={properties.total_memory / 1024**3:.2f} GiB"
    )

assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle settings."
assert n_gpu == 2, "Set the Kaggle accelerator to GPU T4 x2."
assert all("T4" in torch.cuda.get_device_name(i) for i in range(n_gpu))
```

## Cell 2 — Configuration and parent logging

Make global and per-rank batch semantics explicit. The illustrative
configuration below represents global batch 1024 with two rank-local batches
of 512 and no gradient accumulation.

```python
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "input_dir": "/kaggle/input/CONFIRMED-DATASET-SLUG",
    "run_name": "CONFIRMED_RUN_NAME",
    "world_size": 2,
    "distributed_backend": "nccl",
    "global_batch_size": 1024,
    "batch_size_per_rank": 512,
    "gradient_accumulation_steps": 1,
    "epochs": 30,
    "learning_rate": 1e-3,
    "num_workers_per_process": 2,
    "prefetch_factor": 2,
    "omp_num_threads_per_process": 1,
    "seed": 42,
}

assert CONFIG["world_size"] == n_gpu == 2
assert CONFIG["global_batch_size"] == (
    CONFIG["batch_size_per_rank"]
    * CONFIG["world_size"]
    * CONFIG["gradient_accumulation_steps"]
)

INPUT_DIR = Path(CONFIG["input_dir"])
OUT = Path("/kaggle/working") / CONFIG["run_name"]
CACHE = Path("/kaggle/temp") / f"{CONFIG['run_name']}_cache"
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
```

Never deliver the placeholder slug, run name, or illustrative
hyperparameters. Standard logging uses `%` interpolation; format grouped
integers before passing them:

```python
logger.info("Loaded %s rows", f"{row_count:,}")
```

## Notebook launcher

Perform confirmed preprocessing first, close the parent logger, and then launch
the generated entry point. A manifest avoids interpolating complex
configuration into Python source.

```python
DDP_SCRIPT_PATH = CACHE / "training_ddp.py"
MANIFEST_PATH = CACHE / "training_manifest.json"

# DDP_RUNTIME_SOURCE must be a complete ordinary Python module.
DDP_SCRIPT_PATH.write_text(DDP_RUNTIME_SOURCE, encoding="utf-8")
MANIFEST_PATH.write_text(
    json.dumps(
        {
            "config": CONFIG,
            "out_dir": str(OUT),
            "confirmed_data": confirmed_data_manifest,
        },
        indent=2,
    ),
    encoding="utf-8",
)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous_socket:
    rendezvous_socket.bind(("127.0.0.1", 0))
    master_port = rendezvous_socket.getsockname()[1]

command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--nnodes=1",
    "--max-restarts=0",
    f"--nproc-per-node={CONFIG['world_size']}",
    "--master-addr=127.0.0.1",
    f"--master-port={master_port}",
    str(DDP_SCRIPT_PATH),
]
launch_environment = os.environ.copy()
legacy_async_key = "NCCL_" + "ASYNC_ERROR_HANDLING"
launch_environment.pop(legacy_async_key, None)
launch_environment.update(
    {
        "TRAINING_MANIFEST": str(MANIFEST_PATH),
        "OMP_NUM_THREADS": str(CONFIG["omp_num_threads_per_process"]),
        "MKL_NUM_THREADS": str(CONFIG["omp_num_threads_per_process"]),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_DEBUG": "WARN",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
)

logger.info("Launching DDP: %s", " ".join(command))
logging.shutdown()
launch_started = time.perf_counter()
try:
    subprocess.run(command, check=True, env=launch_environment)
except Exception:
    failure = "DDP launcher failed:\n" + traceback.format_exc()
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(failure + "\n")
    print(failure, file=sys.stderr)
    raise
torchrun_wall_seconds = time.perf_counter() - launch_started
```

After the subprocess returns, configure the parent logger again with
`FileHandler(..., mode="a")` before appending output-verification messages.

## DDP entry point — initialization and logging

Keep this code in the generated `.py` module, not directly in a notebook cell.

```python
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

MANIFEST = json.loads(
    Path(os.environ["TRAINING_MANIFEST"]).read_text(encoding="utf-8")
)
CONFIG = MANIFEST["config"]
OUT = Path(MANIFEST["out_dir"])


def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == CONFIG["world_size"] == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=CONFIG["distributed_backend"],
        init_method="env://",
        device_id=device,
    )
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
        )
    ]
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("training_ddp")
```

Seed each rank deterministically. The sampler controls training order, so call
`set_epoch` even with fixed seeds.

```python
random.seed(CONFIG["seed"] + rank)
np.random.seed(CONFIG["seed"] + rank)
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])
torch.backends.cudnn.benchmark = True
```

At the end of the generated module, put orchestration in a guarded entry point
and destroy the process group even after failures:

```python
def main():
    logger = None
    try:
        rank, local_rank, world_size, device = setup_distributed()
        logger = setup_logger(rank)
        run(rank, local_rank, world_size, device, logger)
    except Exception:
        if logger is not None:
            logger.error("DDP rank failed:\n%s", traceback.format_exc())
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## DDP entry point — samplers and loaders

Use `DistributedSampler` only for training. With `drop_last=False`, it may add
up to `world_size - 1` duplicate rows so every rank executes the same number of
steps. Record `sampler.total_size - len(dataset)`.

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(dataset, rank, world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=CONFIG["seed"],
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size_per_rank"],
        sampler=sampler,
        shuffle=False,
        num_workers=CONFIG["num_workers_per_process"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=False,
    )
    return loader, sampler
```

Use an exact strided sampler for validation/test. It neither drops nor repeats
rows and supports different shard lengths because evaluation calls the
unwrapped model, which has no DDP forward collectives.

```python
class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


def make_eval_loader(dataset, rank, world_size):
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size_per_rank"],
        sampler=DistributedEvalSampler(dataset, rank, world_size),
        shuffle=False,
        num_workers=CONFIG["num_workers_per_process"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=False,
    )
```

For datasets smaller than the worker count or when workers are confirmed as
zero, omit `prefetch_factor` and `persistent_workers`.

## DDP entry point — model and AMP

Convert BatchNorm before wrapping only when the confirmed semantics require
global-batch statistics.

```python
model_core = ConfirmedModel(...).to(device)
if confirmed_sync_batchnorm:
    model_core = nn.SyncBatchNorm.convert_sync_batchnorm(model_core)

model = DDP(
    model_core,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
)
optimizer = ConfirmedOptimizer(model.parameters(), ...)
criterion = ConfirmedLoss(...)
scaler = torch.amp.GradScaler("cuda")
```

Do not enable `static_graph=True` unless the model graph and parameter usage are
known to be static. Do not enable `find_unused_parameters=True` speculatively.

## DDP entry point — training

This skeleton handles gradient accumulation without synchronizing intermediate
microbatches. Adapt metric and scheduler logic to the locked task.

```python
from contextlib import nullcontext

history = []
best_metric = None

for epoch in range(1, CONFIG["epochs"] + 1):
    train_sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    local_examples = torch.zeros((), dtype=torch.int64, device=device)
    optimizer_steps = 0

    for step_index, (features, targets) in enumerate(train_loader):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        group_start = (
            step_index // CONFIG["gradient_accumulation_steps"]
        ) * CONFIG["gradient_accumulation_steps"]
        group_size = min(
            CONFIG["gradient_accumulation_steps"],
            len(train_loader) - group_start,
        )
        sync_now = (
            (step_index - group_start + 1) == group_size
        )
        sync_context = nullcontext() if sync_now else model.no_sync()

        with sync_context:
            with torch.amp.autocast("cuda"):
                logits = model(features)
                raw_loss = criterion(logits, targets)
                backward_loss = raw_loss / group_size
            scaler.scale(backward_loss).backward()

        if sync_now:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        batch_examples = targets.size(0)
        local_loss_sum += raw_loss.detach().to(torch.float64) * batch_examples
        local_examples += batch_examples

    dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_examples, op=dist.ReduceOp.SUM)
    train_loss = float(local_loss_sum.item() / max(local_examples.item(), 1))

    validation = evaluate_distributed(
        model.module,
        val_loader,
        len(val_dataset),
        device,
    )
    epoch_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_examples": int(local_examples.item()),
        "optimizer_steps_per_rank": optimizer_steps,
        "sampler_padding_rows": int(
            train_sampler.total_size - len(train_dataset)
        ),
        "validation_loss": validation["loss"],
        **validation["metrics"],
    }

    if rank == 0:
        history.append(epoch_row)
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.module.state_dict().items()
            },
            "epoch": epoch,
            "config": CONFIG,
            "validation_metrics": validation["metrics"],
        }
        torch.save(checkpoint, OUT / "checkpoints" / "last.pt")
        if is_better(validation["metrics"], best_metric):
            best_metric = validation["metrics"]
            torch.save(checkpoint, OUT / "checkpoints" / "best.pt")
        logger.info("Epoch %03d/%03d | %s", epoch, CONFIG["epochs"], epoch_row)
```

All ranks must execute collectives in the same order. Never place a collective
inside a rank-0-only branch. Call NCCL barriers as
`dist.barrier(device_ids=[local_rank])`; do not leave the device implicit.

## DDP entry point — exact distributed evaluation

For classification, reduce raw loss sums, example counts, and the confusion
matrix before deriving metrics. Do not average already-computed rank metrics.

```python
@torch.no_grad()
def evaluate_distributed(model_core, loader, expected_examples, device):
    model_core.eval()
    num_classes = CONFIG["num_classes"]
    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    example_count = torch.zeros((), dtype=torch.int64, device=device)

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model_core(features)
            loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        batch_examples = targets.size(0)
        loss_sum += loss.to(torch.float64) * batch_examples
        example_count += batch_examples
        encoded = targets.to(torch.int64) * num_classes + predictions
        matrix += torch.bincount(
            encoded,
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(example_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
    total = int(example_count.item())
    assert total == expected_examples
    matrix_numpy = matrix.cpu().numpy()
    return {
        "loss": float(loss_sum.item() / max(total, 1)),
        "examples": total,
        "confusion_matrix": matrix_numpy,
        "metrics": metrics_from_confusion(matrix_numpy),
    }
```

For regression, all-reduce sufficient statistics or gather predictions and
targets in bounded chunks. Do not pad evaluation data.

Reload `best.pt` on rank 0, broadcast the unwrapped state tensors to all ranks,
and then run the final test exactly once. Put
`dist.destroy_process_group()` in `finally`.

## Outputs and inline plots

Only rank 0 writes consolidated artifacts:

```text
/kaggle/working/{run_name}/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── run.log
│   └── rank_1.log
├── metrics/
│   ├── config.json
│   ├── history.csv
│   ├── history.json
│   ├── summary.json
│   ├── runtime_breakdown.json
│   └── task-specific reports
└── artifacts/
    └── confirmed diagnostic PNG files
```

The DDP subprocess uses `MPLBACKEND=Agg`. In the next ordinary notebook cell,
verify nonempty required files and display plots:

```python
from IPython.display import Image, display

missing = [str(path) for path in required_outputs if not path.is_file()]
empty = [
    str(path)
    for path in required_outputs
    if path.is_file() and path.stat().st_size == 0
]
assert not missing, f"Missing outputs: {missing}"
assert not empty, f"Empty outputs: {empty}"

for artifact_path in confirmed_artifact_paths:
    display(Image(filename=str(artifact_path)))
```

Record:

- wall-clock preprocessing, training/validation, final evaluation, plotting,
  `torchrun`, and total pipeline time;
- peak CUDA allocated and reserved bytes/MiB per rank;
- logical task communication and DDP gradient all-reduce estimates when
  communication reporting is requested.

## Validation checklist

- The notebook and every ordinary code cell parse.
- The embedded DDP module parses independently.
- The launcher uses `torch.distributed.run`, two processes, zero restarts, and
  an explicit `127.0.0.1` rendezvous with a dynamically selected free port.
- Runtime uses NCCL, `LOCAL_RANK`, one device per process, DDP, AMP,
  device-bound process-group initialization and barriers,
  `DistributedSampler`, and `set_epoch`.
- The child environment removes `NCCL_ASYNC_ERROR_HANDLING`, sets
  `TORCH_NCCL_ASYNC_ERROR_HANDLING`, and pins NCCL bootstrap to loopback.
- Global batch arithmetic matches the approved configuration.
- Evaluation has no padding or duplicate samples.
- Only rank 0 writes consolidated checkpoints and metrics.
- Rank 1 has a separate log.
- Checkpoints contain unwrapped state keys.
- Errors are logged with full tracebacks.
- Lazy logging has no invalid formats such as `%,d`.
- Required outputs are nonempty and history row counts match the configured
  epochs/rounds/clients.
- No placeholders, sample slugs, non-Kaggle runtime paths, or single-process
  multi-GPU wrappers remain.
- Without a real Kaggle T4 x2 run, describe validation as structural only.
