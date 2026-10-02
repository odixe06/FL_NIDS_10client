---
name: kaggle-training-notebook
description: Build structurally validated PyTorch federated-learning benchmark notebooks for exactly two Kaggle NVIDIA T4 GPUs and ten clients. Use for client-parallel GRU, Transformer, or CNN-1D experiments that must apply the locked method-specific local split, checkpoint every round, evaluate prescribed server/client scopes on the shared global test set after every round, and emit the same ten multiclass comparison metrics.
---

# Build a two-T4 federated benchmark notebook

Produce executable `.ipynb` files for a Kaggle `GPU T4 x2` session. Treat the
local workspace as authoring-only. Use `/kaggle/input`, `/kaggle/working`, and
`/kaggle/temp` for runtime input, persistent output, and disposable files.

## Preserve the locked benchmark contract

- Use PyTorch, CUDA AMP, and exactly two NVIDIA T4 GPUs.
- Use ten federated clients and `client_parallel` execution.
- Use the locked method-specific local-data policy: PerFed-SKD alone uses a
  deterministic 90% train / 10% validation split; every other method trains on
  100% of each pre-generated client file without local validation.
- Evaluate the pre-generated `global_test_data.csv` after every round.
- Use exactly the ten classification metrics defined below.
- Save every required server/client checkpoint at every round.
- Do not create `best.pt`, select a round, early-stop, or tune from global-test
  results.
- Use the final round as the fixed comparison endpoint while retaining metrics
  for every round.
- Preserve every confirmed value exactly. Ask one consolidated round of
  questions for unresolved values; never insert a plausible default.
- Capture logs, structured metrics, runtime, logical communication, GPU memory,
  GPU utilization, cache telemetry, and diagnostic plots.
- Keep all configuration in one JSON-serializable `CONFIG` and record execution
  topology and evaluation scope in it.

Repeated global-test evaluation is an explicitly approved experiment contract.
Label it as such in the output metadata. Do not describe the test set as a
single-use holdout and do not use its metrics to choose a checkpoint.

## 1. Pass the specification gate

Read the complete local `data_description.md` and the applicable paper,
methodology note, and `ARCHITECTURE_AND_OUTPUT_SPEC.md`. Confirm all of the
following before generating notebook cells:

- exact `/kaggle/input/...` directory and required filenames;
- target column, ordered features, dtypes, label range, class count, and model
  input/output shape;
- ten client files and the already separated global test file;
- method equations, aggregation boundary, server state, client state, and
  whether personalized client state persists between rounds;
- GRU, Transformer, CNN-1D, proxy, encoder, or auxiliary architecture and
  deterministic initialization contract;
- rounds, method-specific phases, local epochs, per-client batch, optimizer,
  learning rate, loss, scheduler, accumulation, participation, and seeds;
- permitted method-internal search pools or cross-validation. Outside the
  confirmed PerFed-SKD split, full client training and every reported test
  evaluation must never be silently sampled;
- checkpoint contents, evaluation entities, run names, and output paths.

If a method needs internal model selection or reward feedback but no non-test
source is confirmed, stop and ask. Never use the global test set as training
input, distillation input, feature-selection feedback, PPO reward feedback,
hyperparameter tuning data, or an early-stopping signal.

Summarize the locked specification for approval. Only then read the complete
[`references/client_parallel_templates.md`](references/client_parallel_templates.md)
and generate notebooks.

### Locked local-data policies

For `PerFed-SKD` only:

- split each client deterministically and stratified into 90% local train and
  10% local validation;
- keep singleton-class rows in training and preserve at least one training row
  for every represented class;
- use post-update client validation `accuracy` only for the paper's mean-
  accuracy threshold and next-round client selection;
- after every round, record all ten metrics for every client's local validation
  evaluation, while only `accuracy` affects selection;
- do not use validation to create `best.pt`, early-stop, or change the fixed
  ten-round budget.

For FD-IDS, FedCAPS, pFedES, and ProxyModel:

- train on 100% of every client file;
- create no local validation split;
- treat any method-internal evaluator/search data source as a separate
  specification item that requires confirmation and must not use global test.

For FedCAPS specifically, the confirmed search-only evaluator uses
deterministic 5-fold cross-validation on a stratified search pool of at most
100,000 rows per client. This does not replace final full-local training, does
not create a persistent local validation split, and never uses global test.

## 2. Determine evaluation scope

Classify state semantics, not merely architecture names:

### Server-only evaluation

Evaluate only the aggregated server classifier when every participating client
starts each round from the round-start global classifier and no distinct local
classifier state is retained for deployment across rounds. Local states may be
temporary aggregation inputs, but they are not reported as personalized
models.

Save the complete server checkpoint after every round. Do not save redundant
client classifier checkpoints in this case unless the method specification
explicitly requires them for resume.

### Personalized client and server evaluation

When local classifier states persist or otherwise remain distinct after a
round, evaluate every client classifier on the complete global test set after
that round. Also evaluate the server state when it is a classifier that returns
the confirmed class logits.

Save complete checkpoints for all ten clients and every server-side method
state after every round.

### Server state without a classifier

Do not invent a classifier head or central training step. For example, a
feature-selection server or feature-extractor-only proxy cannot produce the ten
classification metrics. Save its complete server state, emit
`server_evaluation_status.json` with `status="not_applicable"` and a precise
reason, and evaluate all persistent client classifiers.

Record the chosen rule and reasoning in `CONFIG`, `summary.json`, and the
checkpoint manifest.

## 3. Use exactly ten comparison metrics

For every evaluable entity and round, derive metrics from one raw 34-class
confusion matrix. Store ratios in `[0,1]` and use zero for a zero denominator.
Let `TP_c`, `FP_c`, `FN_c`, and `support_c` be one-vs-rest counts for class
`c`:

1. `accuracy`
2. `macro_precision`
3. `micro_precision`
4. `weighted_precision`
5. `macro_recall`
6. `micro_recall`
7. `weighted_recall`
8. `macro_f1`
9. `micro_f1`
10. `weighted_f1`

Use:

```text
precision_c = TP_c / (TP_c + FP_c)
recall_c    = TP_c / (TP_c + FN_c)
f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

macro(metric)    = unweighted mean across all confirmed classes
weighted(metric) = sum(support_c * metric_c) / sum(support_c)

micro_precision = sum(TP_c) / (sum(TP_c) + sum(FP_c))
micro_recall    = sum(TP_c) / (sum(TP_c) + sum(FN_c))
micro_f1        = harmonic_mean(micro_precision, micro_recall)
accuracy        = trace(confusion_matrix) / total_examples
```

For single-label multiclass classification, micro precision, micro recall,
micro F1, and accuracy are mathematically equal. Still compute or assign and
store all four named fields independently. Do not add binary attack metrics,
FPR/FNR, or alternative headline classification metrics to the common
comparison schema. Method-specific training losses and operational telemetry
may be stored separately.

## 4. Use client-parallel two-GPU execution

Use one CPU coordinator and exactly two persistent spawned worker processes:

- worker 0 binds to `cuda:0`; worker 1 binds to `cuda:1`;
- do not import or initialize `torch.distributed` and do not use DDP, NCCL,
  `torchrun`, `nn.DataParallel`, `DistributedSampler`, or SyncBatchNorm;
- perform aggregation, consolidated histories, checkpoint manifests, and plots
  only in the coordinator;
- let workers write only unique temporary payloads and dedicated diagnostic
  logs;
- configure each worker logger before CUDA initialization, immediately emit a
  startup record, and record task lifecycle events. Creating a `FileHandler`
  alone only creates a zero-byte file and does not satisfy the logging
  contract;
- propagate tracebacks and nonzero exits; fail on missing or duplicate tasks.

### Balance work without changing batch semantics

- Define batch size per client update on one GPU. Never reinterpret it as a
  batch split across two GPUs.
- Benchmark representative steps for every active model family on both GPUs.
- Estimate client work as
  `ceil(train_examples / per_client_batch) * measured_seconds_per_step`.
- Enumerate nontrivial bipartitions of ten clients and choose the deterministic
  assignment minimizing predicted makespan. Tie-break by client IDs.
- Keep ownership fixed while using GPU-resident caches.
- Record predicted/actual worker load, idle time, and assignment.

### Use VRAM and streams safely

- Cache assigned clients' features, labels, and required state on their owning
  GPU, respecting the confirmed safety fraction, normally 70% of VRAM.
- Treat arrays opened by `np.load(..., mmap_mode="r")` or
  `np.memmap(..., mode="r")` as read-only. Before `torch.from_numpy`, create a
  bounded writable staging copy such as
  `np.array(memmap_slice, copy=True)`. Never pass `np.asarray(read_only_slice)`
  to `torch.from_numpy` and never suppress the non-writable-array warning.
- If data does not fit, use deterministic whole-client LRU plus pinned-memory,
  non-blocking fallback. Never reduce the data.
- Benchmark one versus two independent CUDA streams with real active models
  and the confirmed batch. Select two only on measured throughput improvement
  and only when stochastic-model reproducibility remains valid.
- Give concurrent clients separate models, optimizers, scalers, CUDA streams,
  and RNG derived from `(seed, phase, round, client_id)`.
- Create stream-owned permutations, loss accumulators, and buffers inside the
  owning stream context. Synchronize at phase boundaries, not per batch. Avoid
  `.item()` and CPU reads in batch loops.
- Do not use CUDA Graphs unless explicitly approved.

Keep a frozen cuDNN GRU/LSTM/RNN in `train()` when backward must traverse it to
update an upstream proxy or adapter. Freeze parameters with
`requires_grad_(False)` and clear optimizer gradients; module mode and
trainability are separate concerns.

## 5. Checkpoint every round

Persist under:

```text
/kaggle/working/{run_name}/checkpoints/
├── checkpoint_manifest.json
├── round_001/
│   ├── server.pt          # complete server/method state when it exists
│   ├── client_01.pt       # only for persistent/distinct client models
│   └── ... client_10.pt
├── round_002/
└── ... round_{R:03d}/
```

Do not write `best.pt`. Each round checkpoint must be sufficient to reproduce
or resume its state and include, as applicable:

- `round`, `method`, `scenario`, `model_scope`, and `client_id`;
- model, proxy, encoder/decoder, actor/critic, or other method state;
- optimizer and scaler state when their state persists across rounds;
- deterministic initialization hashes and current state hashes;
- serialized `CONFIG`, ordered features, label mapping, and model metadata;
- RNG/resume metadata and logical communication counters;
- relative path to that entity's per-round global-test metric row.

The manifest must enumerate every expected and written checkpoint, state why
client or server classifier checkpoints are applicable, and verify no round,
client, or server state is missing. A `last.pt` compatibility bundle is allowed
only when a repository contract requires it; it must point to or duplicate the
fixed final round and must not imply model selection.

## 6. Evaluate after every round

- Use the complete `global_test_data.csv` after aggregation and checkpointing
  in every round.
- Never train on it or use its metrics to alter later-round training.
- For one server classifier, shard test indices into two exact, non-overlapping
  GPU partitions and add raw confusion matrices/example counts in the
  coordinator.
- For personalized models, evaluate every client model on the complete global
  test set. Balance model-evaluation tasks across the two fixed workers.
- Assert exact accounting: one server evaluation has `global_test_rows`
  predictions; ten client evaluations have ten independent rows, each with
  exactly `global_test_rows` predictions.
- Save a raw `int64` confusion matrix for every evaluated entity and round so
  all ten metrics can be recomputed.

Write a canonical tidy table to both
`metrics/evaluation_metrics.csv` and `metrics/evaluation_metrics.json`. Each
row contains:

```text
method, scenario, run_name, round, model_scope, client_id,
checkpoint_relative_path, test_examples,
accuracy, macro_precision, micro_precision, weighted_precision,
macro_recall, micro_recall, weighted_recall,
macro_f1, micro_f1, weighted_f1
```

Also write `metrics/final_round_metrics.csv/json` as an exact filter of the
canonical table where `round == configured_rounds`; do not rank rows or select
a best entity.

PerFed-SKD additionally writes `metrics/validation_metrics.csv/json`, with one
row per client per round and the same ten metric fields. It records the exact
90/10 accounting and marks `accuracy` as the sole device-selection field.

## 7. Generate cells in a safe order

1. Assert CUDA and exactly two GPUs whose names contain `T4`; report Python,
   PyTorch, CUDA, GPU names, capabilities, and VRAM.
2. Define serializable `CONFIG`, deterministic seeds, paths, output folders,
   and notebook logging.
3. Validate all confirmed data and create disposable memmaps if needed.
4. Write a self-contained Python entry point under `/kaggle/temp`.
5. Write a JSON manifest.
6. Close notebook logging and launch the entry point as an ordinary subprocess.
7. Reopen logging; verify checkpoints, row counts, confusion matrices, metrics,
   histories, and all required nonempty outputs; display plots inline.

Keep multiprocessing orchestration under `if __name__ == "__main__":`. Never
spawn CUDA workers directly from an ordinary notebook cell.

## 8. Required persistent outputs

Use this common core, adding only method-specific histories/artifacts needed by
the paper:

```text
/kaggle/working/{run_name}/
├── checkpoints/
│   ├── checkpoint_manifest.json
│   └── round_001/ ... round_{R:03d}/
├── logs/
│   ├── run.log
│   ├── worker_0.log
│   └── worker_1.log
├── metrics/
│   ├── config.json
│   ├── dataset_summary.json
│   ├── client_class_distribution.csv
│   ├── evaluation_metrics.csv
│   ├── evaluation_metrics.json
│   ├── final_round_metrics.csv
│   ├── final_round_metrics.json
│   ├── validation_metrics.csv       # PerFed-SKD only
│   ├── validation_metrics.json      # PerFed-SKD only
│   ├── server_evaluation_status.json
│   ├── summary.json
│   ├── communication_costs.json
│   └── runtime_breakdown.json
└── artifacts/
    ├── class_distribution.png
    ├── evaluation_metric_curves.png
    ├── loss_curves.png
    ├── runtime_per_round.png
    ├── communication_cumulative.png
    └── gpu_utilization.png
```

Store per-round/entity confusion matrices under a documented dynamic subtree,
for example `metrics/confusion_matrices/round_001/server.npy` or
`client_01.npy`. Do not require a misleading classification report or best
checkpoint.

For a legacy run that completed before worker success logging was added, do
not remove the worker logs from the required outputs or silently fabricate
task events. First verify `summary.status="complete"`, metrics, checkpoints,
and consolidated runtime telemetry. A clearly labeled recovery log may be
reconstructed only from those existing telemetry fields when preserving the
completed run is explicitly preferred; state that original per-worker events
are unavailable. Re-run the fixed notebook when authentic worker event logs
are required.

## 9. Validate before handoff

Run:

```bash
python scripts/validate_client_parallel_notebook.py notebook.ipynb
```

In addition to notebook JSON and Python syntax, verify:

- exact two-T4 assertion, client-parallel topology, AMP, fixed per-client batch,
  deterministic initialization, workload assignment, cache, and stream safety;
- CPU validation of feature shape/dtype, integer class IDs, and full client
  sample accounting before CUDA transfer;
- exact split policy: PerFed-SKD 90/10 with validation accuracy for selection;
  every other method 100% local train with no validation split;
- global-test evaluation after every round with no feedback into training;
- exact ten-metric names, formulas, ranges, entity counts, and final-round
  filter;
- all required server/client round checkpoints and manifest entries;
- absence of `best.pt`, early stopping, test-based selection, DDP, NCCL, and
  unsafe consolidated worker writes;
- required logs, metrics, runtime, communication, memory, utilization, and
  plots are nonempty;
- worker startup and task records make both worker logs nonempty on successful
  runs; exception-only logging is invalid;
- every read-only memmap slice is copied to a writable bounded host buffer
  before `torch.from_numpy`, with no warning suppression;
- worker exception propagation and exact output/history row counts.

Report validation as structural unless the notebook actually completed on
Kaggle `GPU T4 x2`. Hand off paths, locked specification, validation performed,
and any explicitly approved deviations.
