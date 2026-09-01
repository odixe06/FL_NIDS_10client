---
name: kaggle-training-notebook
description: Build, regenerate, fix, or review the locked task10 federated-learning benchmark notebooks that run on a Kaggle GPU T4 x2 session — ten clients, client-parallel two-worker execution, per-round checkpoints, per-round global-test evaluation, and the same ten multiclass metrics for every method. Use for any GRU, Transformer, or CNN-1D notebook belonging to FD-IDS, PerFed-SKD, FedCAPS, pFedES, or ProxyModel.
---

# Two-T4 federated benchmark notebooks

You author `.ipynb` files here; they execute on Kaggle. The local workspace is
**authoring-only** — never treat a local path as a runtime path. At runtime the
notebook reads `/kaggle/input`, persists to `/kaggle/working`, and writes
disposable files to `/kaggle/temp`.

Every method in task10 reconstructs a different paper, but all five share one
locked benchmark contract so their numbers are comparable. Your job is to keep
that contract exact. A notebook that trains correctly but breaks the contract is
a failed deliverable.

## Non-negotiables

Hold these true at every step. They are not defaults to be weighed — they are the
contract.

- PyTorch, CUDA AMP, and **exactly two NVIDIA T4 GPUs**.
- **Ten** federated clients, `client_parallel` execution.
- The locked method-specific local-data policy: **PerFed-SKD alone** uses a
  deterministic 90% train / 10% validation split; **every other method** trains
  on 100% of each pre-generated client file with no local validation.
- Evaluate the pre-generated `global_test_data.csv` **after every round**.
- Exactly the **ten** classification metrics defined below — no more, no fewer.
- Save **every** required server/client checkpoint at **every** round.
- Never create `best.pt`, select a round, early-stop, or tune from global-test
  results.
- The **final round is the fixed comparison endpoint**; still retain metrics for
  every round.
- Preserve every confirmed value **exactly**. Ask one consolidated round of
  questions for unresolved values; never insert a plausible default.
- Capture logs, structured metrics, runtime, logical communication, GPU memory,
  GPU utilization, cache telemetry, and diagnostic plots.
- Keep all configuration in one JSON-serializable `CONFIG`, and record execution
  topology and evaluation scope inside it.

Repeated global-test evaluation is an **explicitly approved** experiment
contract. Label it as such in the output metadata. Do not describe the test set
as a single-use holdout, and do not use its metrics to choose a checkpoint.

## Where things live

Each paper folder under `nckh/task10/` has the same shape:

```text
<paper_folder>/
├── <paper>.md                        # the source paper
├── data_description.md               # dataset spec
├── ARCHITECTURE_AND_OUTPUT_SPEC.md   # this method's locked deltas
├── 10_clients_gru/                   # one notebook each
├── 10_clients_transformer/
├── 10_clients_cnn1d/
└── scripts/
    ├── build_*_notebooks.py          # notebook builder for this method
    └── *_runtime.py                  # the embedded client-parallel runtime
```

The shared contract lives one level up at `../ARCHITECTURE_AND_OUTPUT_SPEC.md`,
and four of the five methods generate their notebooks through the shared builder
`../scripts/build_common_notebooks.py --method <method>` over
`../scripts/common_client_parallel_runtime.py`. FedCAPS
(`permutation_feature_importance`) is the exception: it has a **standalone**
builder and runtime under its own `scripts/`.

So notebooks in this repo are **generated, not hand-edited**. When a notebook
must change, change the builder or the runtime module and re-run the builder:

```bash
python3 scripts/build_<method>_notebooks.py    # from inside the paper folder
```

This machine has `python3` only — `python` is not on `PATH`.

Editing a generated `.ipynb` directly is acceptable only for a one-off
investigation you intend to throw away — say so explicitly if you do it, because
the next builder run overwrites it.

## Gate 1 — Read the specification

Write no cell, and change no builder line, until you have read in full:

- `data_description.md`;
- the applicable paper `.md` in the folder, plus any methodology note;
- `ARCHITECTURE_AND_OUTPUT_SPEC.md` **and** the shared
  `../ARCHITECTURE_AND_OUTPUT_SPEC.md`.

Read them completely with the Read tool. Do not skim, and do not rely on what a
similar method did — the deltas between methods are exactly where the contract
breaks.

**Completion criterion:** you can state, without guessing, every item in the
Gate 2 checklist below. Anything you cannot state becomes a question, not an
assumption.

## Gate 2 — Lock the specification

Confirm a concrete value for each item. Ask about all unresolved items in **one
consolidated round** of questions.

- exact `/kaggle/input/...` directory and required filenames;
- target column, ordered features, dtypes, label range, class count, and model
  input/output shape;
- ten client files and the already separated global test file;
- method equations, aggregation boundary, server state, client state, and
  whether personalized client state persists between rounds;
- GRU, Transformer, CNN-1D, proxy, encoder, or auxiliary architecture and the
  deterministic initialization contract;
- rounds, method-specific phases, local epochs, per-client batch, optimizer,
  learning rate, loss, scheduler, accumulation, participation, and seeds;
- permitted method-internal search pools or cross-validation. Outside the
  confirmed PerFed-SKD split, full client training and every reported test
  evaluation must **never** be silently sampled;
- checkpoint contents, evaluation entities, run names, and output paths.

**Stop and ask** if a method needs internal model selection or reward feedback
but no non-test source is confirmed. The global test set is never training
input, distillation input, feature-selection feedback, PPO reward feedback,
hyperparameter tuning data, or an early-stopping signal.

Then summarize the locked specification and get approval.

**Completion criterion:** every item above has a confirmed value, the user has
approved the summary, and no placeholder or "I'll assume" remains.

### Locked local-data policy

**PerFed-SKD only:**

- split each client deterministically and stratified into 90% local train and
  10% local validation;
- keep singleton-class rows in training, and preserve at least one training row
  for every represented class;
- use post-update client validation `accuracy` **only** for the paper's
  mean-accuracy threshold and next-round client selection;
- after every round, record all ten metrics for every client's local validation
  evaluation, while only `accuracy` affects selection;
- do not use validation to create `best.pt`, early-stop, or change the fixed
  ten-round budget.

**FD-IDS, FedCAPS, pFedES, ProxyModel:**

- train on 100% of every client file;
- create no local validation split;
- treat any method-internal evaluator/search data source as a separate
  specification item that requires confirmation and must not use global test.

**FedCAPS specifically:** the confirmed search-only evaluator uses deterministic
5-fold cross-validation on a stratified search pool of at most 100,000 rows per
client. This does not replace final full-local training, does not create a
persistent local validation split, and never touches global test.

## Gate 3 — Decide the evaluation scope

Classify **state semantics**, not architecture names. Pick exactly one rule.

**Server-only evaluation.** Use when every participating client starts each
round from the round-start global classifier and no distinct local classifier
state is retained for deployment across rounds. Local states may be temporary
aggregation inputs, but they are not reported as personalized models.

→ Save the complete server checkpoint after every round. Do not save redundant
client classifier checkpoints unless the method specification explicitly
requires them for resume.

**Personalized client and server evaluation.** Use when local classifier states
persist or otherwise remain distinct after a round. Evaluate every client
classifier on the complete global test set after that round, and also evaluate
the server state when it is a classifier returning the confirmed class logits.

→ Save complete checkpoints for all ten clients and every server-side method
state after every round.

**Server state without a classifier.** Do not invent a classifier head or a
central training step. A feature-selection server or a feature-extractor-only
proxy cannot produce the ten classification metrics.

→ Save its complete server state, emit `server_evaluation_status.json` with
`status="not_applicable"` and a precise reason, and evaluate all persistent
client classifiers.

Record the chosen rule and the reasoning in `CONFIG`, `summary.json`, and the
checkpoint manifest.

**Completion criterion:** the scope is one of the three rules above, justified by
state semantics, and it is written into `CONFIG`.

## The ten comparison metrics

For every evaluable entity and round, derive metrics from **one raw 34-class
confusion matrix**. Store ratios in `[0,1]`; a zero denominator yields zero.

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

With `TP_c`, `FP_c`, `FN_c`, `support_c` as one-vs-rest counts for class `c`:

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

For single-label multiclass, micro precision, micro recall, micro F1, and
accuracy are mathematically equal. Still compute or assign and store all four
named fields independently.

Do **not** add binary attack metrics, FPR/FNR, or alternative headline
classification metrics to the common comparison schema. Method-specific training
losses and operational telemetry belong in separate files.

## Gate 4 — Generate the notebook

Read [`references/client-parallel-runtime.md`](references/client-parallel-runtime.md)
in full before writing runtime code, and
[`references/outputs-and-metrics.md`](references/outputs-and-metrics.md) before
writing checkpoint, evaluation, or output-verification code. Read them only now
— reading them before Gate 2 passes invites you to code around an unconfirmed
spec.

### Execution topology

One CPU coordinator and exactly two persistent spawned worker processes:

- worker 0 binds to `cuda:0`; worker 1 binds to `cuda:1`;
- do **not** import or initialize `torch.distributed`, and do not use DDP, NCCL,
  `torchrun`, `nn.DataParallel`, `DistributedSampler`, or SyncBatchNorm;
- perform aggregation, consolidated histories, checkpoint manifests, and plots
  **only** in the coordinator;
- workers write only unique temporary payloads and dedicated diagnostic logs;
- configure each worker logger **before** CUDA initialization, immediately emit
  a startup record, and record task lifecycle events. Creating a `FileHandler`
  alone leaves a zero-byte file and does **not** satisfy the logging contract;
- propagate tracebacks and nonzero exits; fail on missing or duplicate tasks.

### Balance work without changing batch semantics

- Batch size is defined **per client update on one GPU**. Never reinterpret it
  as a batch split across two GPUs.
- Benchmark representative steps for every active model family on both GPUs.
- Estimate client work as
  `ceil(train_examples / per_client_batch) * measured_seconds_per_step`.
- Enumerate nontrivial bipartitions of ten clients and choose the deterministic
  assignment minimizing predicted makespan. Tie-break by client IDs.
- Keep ownership fixed while using GPU-resident caches.
- Record predicted/actual worker load, idle time, and assignment.

### Use VRAM and streams safely

- Cache assigned clients' features, labels, and required state on their owning
  GPU, respecting the confirmed safety fraction — normally 70% of VRAM.
- Treat arrays opened by `np.load(..., mmap_mode="r")` or
  `np.memmap(..., mode="r")` as **read-only**. Before `torch.from_numpy`, create
  a bounded writable staging copy such as `np.array(memmap_slice, copy=True)`.
  Never pass `np.asarray(read_only_slice)` to `torch.from_numpy`, and never
  suppress the non-writable-array warning.
- If the data does not fit, use deterministic whole-client LRU plus
  pinned-memory, non-blocking fallback. **Never reduce the data.**
- Benchmark one versus two independent CUDA streams with real active models and
  the confirmed batch. Select two only on measured throughput improvement and
  only when stochastic-model reproducibility remains valid.
- Give concurrent clients separate models, optimizers, scalers, CUDA streams,
  and RNG derived from `(seed, phase, round, client_id)`.
- Create stream-owned permutations, loss accumulators, and buffers inside the
  owning stream context. Synchronize at phase boundaries, not per batch. Avoid
  `.item()` and CPU reads in batch loops.
- Do not use CUDA Graphs unless explicitly approved.

Keep a frozen cuDNN GRU/LSTM/RNN in `train()` when backward must traverse it to
update an upstream proxy or adapter. Freeze parameters with
`requires_grad_(False)` and clear optimizer gradients — module mode and
trainability are separate concerns.

### Cell order

1. Assert CUDA and exactly two GPUs whose names contain `T4`; report Python,
   PyTorch, CUDA, GPU names, capabilities, and VRAM.
2. Define the serializable `CONFIG`, deterministic seeds, paths, output folders,
   and notebook logging.
3. Validate all confirmed data and create disposable memmaps if needed.
4. Write a self-contained Python entry point under `/kaggle/temp`.
5. Write a JSON manifest.
6. Close notebook logging and launch the entry point as an ordinary subprocess.
7. Reopen logging; verify checkpoints, row counts, confusion matrices, metrics,
   histories, and all required nonempty outputs; display plots inline.

Keep multiprocessing orchestration under `if __name__ == "__main__":`. **Never
spawn CUDA workers directly from an ordinary notebook cell.**

**Completion criterion:** the notebook runs top to bottom in this order, reads
and writes only Kaggle paths, carries no placeholder values, and produces every
output listed in `references/outputs-and-metrics.md`.

## Gate 5 — Validate and hand off

Run the structural validator from the paper folder:

```bash
python3 .claude/skills/kaggle-training-notebook/scripts/validate_client_parallel_notebook.py \
    10_clients_cnn1d/<method>_10_clients_cnn1d.ipynb
```

Pass several notebooks at once when you regenerate the whole method. Beyond
notebook JSON and Python syntax, confirm:

- exact two-T4 assertion, client-parallel topology, AMP, fixed per-client batch,
  deterministic initialization, workload assignment, cache, and stream safety;
- CPU validation of feature shape/dtype, integer class IDs, and full client
  sample accounting before CUDA transfer;
- exact split policy — PerFed-SKD 90/10 with validation accuracy for selection,
  every other method 100% local train with no validation split;
- global-test evaluation after every round with no feedback into training;
- exact ten-metric names, formulas, ranges, entity counts, and final-round
  filter;
- all required server/client round checkpoints and manifest entries;
- absence of `best.pt`, early stopping, test-based selection, DDP, NCCL, and
  unsafe consolidated worker writes;
- required logs, metrics, runtime, communication, memory, utilization, and plots
  are nonempty;
- worker startup and task records make both worker logs nonempty on successful
  runs — exception-only logging is invalid;
- every read-only memmap slice is copied to a writable bounded host buffer
  before `torch.from_numpy`, with no warning suppression;
- worker exception propagation and exact output/history row counts.

**Report validation as structural** unless the notebook actually completed on a
Kaggle `GPU T4 x2` session. Do not imply a real run happened.

Hand off: the paths you touched, the locked specification, the validation you
performed, and any explicitly approved deviations.

## Reference files

- [`references/client-parallel-runtime.md`](references/client-parallel-runtime.md)
  — coordinator/worker topology, launcher, GPU cache, CUDA streams, round
  coordination, and the code patterns that satisfy them.
- [`references/outputs-and-metrics.md`](references/outputs-and-metrics.md)
  — checkpoint layout and contents, per-round global-test evaluation mechanics,
  canonical metric tables, and the complete required output tree.
