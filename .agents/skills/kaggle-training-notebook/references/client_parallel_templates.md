# PyTorch client-parallel federated benchmarking on Kaggle 2×T4

Read this complete reference only after the experiment specification is
approved. Adapt identifiers, method state, and phases without changing the
locked data, batch, evaluation, or checkpoint semantics.

## Contents

- [Topology](#topology)
- [Launcher](#launcher)
- [Worker lifecycle](#worker-lifecycle)
- [Full-data contract](#full-data-contract)
- [Workload partition](#workload-partition)
- [GPU-resident cache](#gpu-resident-cache)
- [CUDA-stream benchmark](#cuda-stream-benchmark)
- [Round coordination](#round-coordination)
- [Checkpointing](#checkpointing)
- [Per-round global-test evaluation](#per-round-global-test-evaluation)
- [Ten-metric computation](#ten-metric-computation)
- [Validation](#validation)

## Topology

Use one ordinary entry-point process as the CPU coordinator. Spawn exactly two
persistent workers, each with its own input queue:

```text
notebook kernel
  └─ training_entry.py (CPU coordinator; only consolidated writer)
       ├─ worker gpu_id=0 → cuda:0
       └─ worker gpu_id=1 → cuda:1
```

Do not use `torch.distributed`, DDP, NCCL, `torchrun`, `nn.DataParallel`,
`DistributedSampler`, or SyncBatchNorm. The notebook may inspect both GPUs
before launching. The coordinator must not create a CUDA context before
spawning workers.

## Launcher

Write the self-contained entry point and JSON manifest under `/kaggle/temp`,
close the notebook logger, and launch an ordinary subprocess:

```python
command = [sys.executable, str(CLIENT_PARALLEL_SCRIPT_PATH)]
launch_environment = os.environ.copy()
launch_environment.update(
    {
        "TRAINING_MANIFEST": str(MANIFEST_PATH),
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
)
subprocess.run(command, check=True, env=launch_environment)
```

Do not set NCCL environment variables.

## Worker lifecycle

Keep worker creation inside the generated module's guarded main:

```python
def worker_main(gpu_id, task_queue, result_queue, manifest_path):
    logger = None
    try:
        logger = configure_worker_logger(
            output_dir / "logs" / f"worker_{gpu_id}.log",
            f"worker_{gpu_id}",
        )
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
        logger.info("Worker %s started on %s", gpu_id, device)
        # Initialize cache, streams, and model ownership.
        while True:
            task = task_queue.get()
            logger.info(
                "Starting task %s (%s)", task["task_id"], task["kind"]
            )
            # Execute the task and report its result.
    except Exception:
        if logger is not None:
            logger.exception("Worker failed")
        result_queue.put(
            {
                "kind": "worker_error",
                "gpu_id": gpu_id,
                "traceback": traceback.format_exc(),
            }
        )
        raise


def main():
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(2)]
    workers = [
        context.Process(
            target=worker_main,
            args=(gpu_id, task_queues[gpu_id], result_queue, manifest_path),
        )
        for gpu_id in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        coordinate(workers, task_queues, result_queue)
    finally:
        # Request shutdown, join, terminate only a worker that failed to exit,
        # and reject every nonzero exit code.
        ...


if __name__ == "__main__":
    main()
```

Use stable task IDs. Expect exactly one success result per task. Treat a
duplicate, missing result, timeout, `worker_error`, or nonzero exit as fatal.
Workers write only unique temporary payloads and `worker_{gpu_id}.log`.
A `FileHandler` creates the path but leaves it at zero bytes until a record is
emitted. Log worker startup before entering the task loop and log each task so
a successful run always produces useful, nonempty diagnostics. Do not rely on
exception-only logging. Logging handlers flush each emitted record; close them
during orderly shutdown when the runtime manages handlers explicitly.

## Full-data contract

- For PerFed-SKD only, deterministically stratify every client file into 90%
  train and 10% validation. Keep singleton classes in train and preserve at
  least one training row per represented class. Record and assert disjoint,
  exhaustive split accounting.
- For every other method, load every row of every confirmed
  `client_{id}_train.csv` for training, create no validation indices, and
  assert `train_rows == source_rows`.
- In PerFed-SKD, evaluate all ten metrics on each client's validation split
  after its local update. Use only validation `accuracy` in the paper's mean-
  accuracy threshold and next-round client selection.
- Keep `global_test_data.csv` in a separate read-only evaluation path.
- Do not expose global-test batches or metrics to training, aggregation,
  feature-selection feedback, PPO reward calculation, schedulers, or stopping.
- A confirmed method-specific search pool may limit only its search phase. It
  must not limit final client training or reported global-test evaluation.
- FedCAPS uses deterministic 5-fold cross-validation only inside its stratified
  search pool of at most 100,000 rows per client. Final personalized training
  still consumes 100% of every client file and has no persistent validation
  split.

Validate cached classification arrays on CPU before CUDA transfer: features
must have the exact confirmed shape and dtype; labels must be integer IDs in
the confirmed range; row indices must be exact, unique where expected, and in
range.

## Workload partition

Benchmark each active model family after CUDA warmup. Measure with CUDA events
and one final phase synchronization, not a CPU timer around asynchronous
kernels.

For client `k`:

```python
steps_k = math.ceil(train_examples_k / per_client_batch_size)
estimated_work_k = steps_k * seconds_per_step[model_family_k]
```

For ten clients, enumerate masks `1 .. 2**10 - 2`. Normalize each partition so
the lexicographically smaller tuple is GPU 0. Minimize:

```text
(max(load_0, load_1), abs(load_0 - load_1), gpu_0_client_tuple)
```

If GPU benchmark rates differ, minimize predicted completion time using each
GPU's measured family-specific rate. Keep ownership fixed while its data or
persistent model state is resident. Record predicted/actual load and idle time.

## GPU-resident cache

Calculate bytes before allocation:

```python
cache_budget_bytes = int(total_vram_bytes * cache_fraction)
```

Use `cache_fraction=0.70` only when confirmed. Count features, labels,
persistent models, method states, and a conservative activation/optimizer
reserve. Load memmaps in bounded chunks into preallocated device tensors.

Read-only memmaps cannot be passed directly to `torch.from_numpy`. Stage each
bounded slice in writable host memory before copying to the GPU:

```python
target = torch.empty(array.shape, dtype=torch_dtype, device=device)
for start in range(0, len(array), chunk_rows):
    end = min(start + chunk_rows, len(array))
    writable = np.array(array[start:end], copy=True)
    source = torch.from_numpy(writable)
    target[start:end].copy_(source, non_blocking=False)
```

Do not replace the writable copy with `np.asarray(array[start:end])`; a view of
`np.load(..., mmap_mode="r")` or `np.memmap(..., mode="r")` remains read-only
and causes PyTorch's non-writable-tensor warning and undefined behavior if the
tensor is ever mutated. Do not hide this warning with a warnings filter.

Cache only assigned clients. Record planned/actual bytes, allocated/reserved
peaks, hit/miss/eviction counts, and full-cache or fallback mode. If data does
not fit, use deterministic whole-client LRU and a DataLoader with
`pin_memory=True`, copying with `non_blocking=True`. Never reduce the data or
call `.pin_memory()` inside a batch loop.

## CUDA-stream benchmark

Benchmark `stream_candidates=[1, 2]` using the confirmed batch and real active
model families. Give each candidate independent models, optimizers, scalers,
batches, and streams. Warm up, then time with CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# Enqueue representative work.
end.record()
torch.cuda.synchronize(device)
elapsed_seconds = start.elapsed_time(end) / 1000.0
```

Choose two streams only when aggregate samples/second improves by the confirmed
margin and deterministic stochastic-model behavior remains valid. Record raw
measurements and the choice.

During real multi-stream work:

- keep separate model/method state, optimizer, scaler, RNG, loss accumulator,
  and stream per concurrent client;
- seed from `(base_seed, phase, round, client_id)`, not GPU or completion order;
- after waiting for any prior default-stream setup, create permutations and
  accumulators inside the owning stream:

```python
client_stream.wait_stream(torch.cuda.current_stream(device))
with torch.cuda.stream(client_stream):
    order = torch.randperm(
        train_examples,
        generator=client_generator,
        device=device,
    )
    loss_sums = torch.zeros(loss_width, dtype=torch.float64, device=device)
```

- never enqueue new default-stream setup after that wait and consume it from
  the client stream without another dependency;
- enqueue batches round-robin and synchronize only at phase/task boundaries;
- do not call `.item()`, `.cpu()`, or global synchronization per batch;
- never share mutable client/proxy parameters between concurrent clients.

Do not use CUDA Graphs unless explicitly approved.

## Round coordination

For each configured round:

1. The coordinator snapshots all round-start server/method state.
2. It dispatches independent owned-client tasks to both persistent workers.
3. Workers train every client on its locked local-train rows—90% for
   PerFed-SKD, 100% otherwise—according to the paper and return unique temporary
   payloads.
4. The coordinator verifies exact task/client/sample accounting.
5. It performs the method-defined aggregation on CPU after all required client
   updates finish.
6. It saves every required server/client checkpoint for this round.
7. For PerFed-SKD, it first records ten validation metrics per client and uses
   only validation accuracy for the paper-defined next-round selection.
8. It evaluates all required server/client classifiers on the complete global
   test set.
9. It appends the canonical metric rows and continues to the next round without
   using evaluation results to alter training.

For server-only methods, discard temporary local classifier states after
aggregation unless needed to resume an in-progress round. For personalized
methods, retain each client's state across rounds and save it after each round.

Normal BatchNorm receives the full per-client batch. When backward must pass
through a frozen cuDNN GRU/LSTM/RNN to update an upstream proxy, keep the module
in `train()` and freeze only parameters with `requires_grad_(False)`.

## Checkpointing

Write checkpoints only from the coordinator:

```text
checkpoints/
├── checkpoint_manifest.json
├── round_001/
│   ├── server.pt
│   ├── client_01.pt
│   └── ...
└── round_{R:03d}/
```

Applicability rules:

- server-only classifier method: one complete `server.pt` per round;
- persistent personalized method: ten complete `client_XX.pt` files per round;
- any method with server-side state: save `server.pt`, even when that state is
  not a classifier and cannot be evaluated;
- never write `best.pt` or select a checkpoint using global-test metrics.

Use atomic temporary-file replacement inside the output directory. Update the
manifest only after every expected file for the round is present and readable.
Record scope, client ID, round, hashes, state inventory, and metric-row path.

## Per-round global-test evaluation

For one server classifier, split exact global-test indices into two
non-overlapping shards. Evaluate one shard per GPU and return raw confusion
matrices and example counts; the coordinator adds them before metrics.

For personalized methods, every client classifier predicts the complete global
test set independently. Assign model-evaluation tasks across the two workers.
Do not split one client's metric computation into a partial support without
recombining its raw confusion matrix first.

Expected evaluation rows for `R` rounds:

```text
server-only classifier                         = R
10 personalized clients, no server classifier = 10 * R
10 personalized clients plus server classifier = 11 * R
```

Save each raw confusion matrix as `int64` under a deterministic round/entity
path. Assert its sum equals `global_test_rows` for that row.

If server state does not return class logits, emit:

```json
{
  "status": "not_applicable",
  "reason": "<precise method-specific reason>",
  "evaluated_clients": 10
}
```

Do not attach a new classifier merely to produce a server metric.

## Ten-metric computation

Compute all metrics centrally from the raw confusion matrix. Use all confirmed
classes in macro means, including classes with zero support, and define zero
denominators as zero.

The canonical row keys are exactly:

```python
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
```

Verify every value is finite and in `[0,1]`. In single-label multiclass runs,
assert micro precision, micro recall, micro F1, and accuracy agree within a
documented floating-point tolerance while still writing all four fields.

Append rows to `evaluation_metrics.csv/json` in deterministic order:
`round`, server before clients when applicable, then ascending `client_id`.
After the final round, filter—not rank—the rows to
`final_round_metrics.csv/json`.

## Validation

Run:

```bash
python scripts/validate_client_parallel_notebook.py notebook.ipynb
```

Also verify:

- exactly two persistent spawned workers and explicit CUDA binding;
- no distributed/DDP/NCCL/DataParallel/SyncBatchNorm usage;
- serialized config locks the method-specific split, per-round global-test
  evaluation, ten metric names, and no best checkpoint;
- PerFed-SKD alone uses exhaustive deterministic 90/10 client splits and only
  validation accuracy for device selection; every other method uses 100% local
  training without validation;
- deterministic exact bipartition, cache budget/fallback, memory telemetry,
  and recorded stream benchmark;
- stream-owned CUDA permutations/accumulators and no per-batch CPU sync;
- aggregation and persistent output writing occur only in the coordinator;
- checkpoint manifest completeness for every applicable entity and round;
- exact global-test accounting and raw confusion matrix sum per metric row;
- exact evaluation row count for the chosen scope;
- final-round files are exact filters of the all-round table;
- global-test results cannot influence training, scheduling, aggregation, or
  checkpoint selection;
- worker exception propagation and clean shutdown;
- a successful worker emits startup and task records, rather than leaving a
  zero-byte log whose only possible record is an exception;
- read-only memmap slices use bounded `copy=True` staging before
  `torch.from_numpy`, and the runtime does not suppress the warning;
- all required outputs are nonempty.
