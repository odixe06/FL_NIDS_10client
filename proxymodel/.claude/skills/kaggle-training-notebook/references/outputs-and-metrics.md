# Checkpoints, evaluation, and required outputs

Read this before writing any checkpoint, evaluation, or output-verification
code. The rules here are what make five different papers comparable, so they are
identical across methods except where a method is named explicitly.

## Contents

- [Checkpointing](#checkpointing)
- [Per-round global-test evaluation](#per-round-global-test-evaluation)
- [Ten-metric computation](#ten-metric-computation)
- [Canonical metric tables](#canonical-metric-tables)
- [Required persistent outputs](#required-persistent-outputs)
- [Legacy runs](#legacy-runs)

## Checkpointing

Write checkpoints **only from the coordinator**, under:

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

Applicability follows the evaluation scope decided in `SKILL.md` Gate 3:

- **server-only classifier method** — one complete `server.pt` per round;
- **persistent personalized method** — ten complete `client_XX.pt` files per
  round;
- **any method with server-side state** — save `server.pt` even when that state
  is not a classifier and cannot be evaluated;
- **never** write `best.pt`, and never select a checkpoint using global-test
  metrics.

### Checkpoint contents

Each round checkpoint must be sufficient to reproduce or resume its state. As
applicable, include:

- `round`, `method`, `scenario`, `model_scope`, and `client_id`;
- model, proxy, encoder/decoder, actor/critic, or other method state;
- optimizer and scaler state when their state persists across rounds;
- deterministic initialization hashes and current state hashes;
- the serialized `CONFIG`, ordered features, label mapping, and model metadata;
- RNG/resume metadata and logical communication counters;
- the relative path to that entity's per-round global-test metric row.

### Manifest

`checkpoint_manifest.json` must enumerate every expected and every written
checkpoint, state **why** client or server classifier checkpoints are
applicable, and verify that no round, client, or server state is missing.

Use atomic temporary-file replacement inside the output directory, and update
the manifest only after every expected file for the round is present and
readable.

A `last.pt` compatibility bundle is allowed only when a repository contract
requires it. It must point to or duplicate the fixed final round, and it must
not imply model selection.

## Per-round global-test evaluation

Evaluate the **complete** `global_test_data.csv` after aggregation and
checkpointing in every round. Never train on it, and never let its metrics alter
later-round training.

**One server classifier:** shard the test indices into two exact,
non-overlapping GPU partitions, evaluate one shard per GPU, return raw confusion
matrices and example counts, and add them in the coordinator before deriving
metrics.

**Personalized models:** every client classifier predicts the complete global
test set independently. Balance the model-evaluation tasks across the two fixed
workers. Do not split one client's metric computation into a partial support
without recombining its raw confusion matrix first.

### Exact accounting

Assert it, do not assume it:

- one server evaluation produces `global_test_rows` predictions;
- ten client evaluations produce ten independent rows, each with exactly
  `global_test_rows` predictions;
- each saved confusion matrix sums to `global_test_rows` for its row.

Expected evaluation rows for `R` rounds:

```text
server-only classifier                        = R
10 personalized clients, no server classifier = 10 * R
10 personalized clients plus server classifier = 11 * R
```

Save every raw confusion matrix as `int64` under a deterministic round/entity
path so all ten metrics can be recomputed later.

### When the server has no classifier

Emit `metrics/server_evaluation_status.json`:

```json
{
  "status": "not_applicable",
  "reason": "<precise method-specific reason>",
  "evaluated_clients": 10
}
```

Do not attach a new classifier merely to produce a server metric.

## Ten-metric computation

Compute all metrics centrally from the raw confusion matrix. Include **all**
confirmed classes in macro means, including classes with zero support, and
define a zero denominator as zero.

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
assert that micro precision, micro recall, micro F1, and accuracy agree within a
documented floating-point tolerance — while still writing all four fields
independently.

## Canonical metric tables

Write one tidy table to **both** `metrics/evaluation_metrics.csv` and
`metrics/evaluation_metrics.json`. Each row contains:

```text
method, scenario, run_name, round, model_scope, client_id,
checkpoint_relative_path, test_examples,
accuracy, macro_precision, micro_precision, weighted_precision,
macro_recall, micro_recall, weighted_recall,
macro_f1, micro_f1, weighted_f1
```

Append rows in deterministic order: by `round`, then the server before clients
when applicable, then ascending `client_id`.

Also write `metrics/final_round_metrics.csv` and `.json` as an **exact filter**
of the canonical table where `round == configured_rounds`. Filter — do not rank,
and do not select a best entity.

**PerFed-SKD additionally** writes `metrics/validation_metrics.csv` and `.json`,
one row per client per round with the same ten metric fields. It records the
exact 90/10 accounting and marks `accuracy` as the sole device-selection field.

## Required persistent outputs

Use this common core, adding only the method-specific histories and artifacts
the paper needs:

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

Store per-round and per-entity confusion matrices under a documented dynamic
subtree — for example `metrics/confusion_matrices/round_001/server.npy` or
`metrics/confusion_matrices/round_001/client_01.npy`.

Do not require a misleading classification report or a best checkpoint.

The exact per-method output additions and expected row counts are in this
folder's `ARCHITECTURE_AND_OUTPUT_SPEC.md` and the shared
`../ARCHITECTURE_AND_OUTPUT_SPEC.md`. Those files are the source of truth for
per-method deltas; do not re-derive them from memory.

The final notebook cell verifies that every required file exists and is
nonempty, that metric row counts and checkpoint counts match the configured
rounds and entities, and that the server classification status is correctly
applicable or not applicable — then displays the plots inline.

## Legacy runs

For a run that completed before worker success logging was added, do **not**
remove the worker logs from the required outputs and do **not** fabricate task
events.

First verify `summary.status="complete"`, plus metrics, checkpoints, and
consolidated runtime telemetry. A clearly labeled recovery log may then be
reconstructed **only** from those existing telemetry fields, and only when
preserving the completed run is explicitly preferred. State plainly that the
original per-worker events are unavailable.

Re-run the fixed notebook when authentic worker event logs are required.
