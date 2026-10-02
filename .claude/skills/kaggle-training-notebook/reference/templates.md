# PyTorch on Kaggle 2×T4 — cell templates

Reference code for the ten cells in Step 3. Adapt the marked spots to the locked spec; keep the multi-GPU + AMP structure intact — that is what exploits both T4s.

## Kaggle environment facts

- **Input (read-only):** `/kaggle/input/<dataset-slug>/...` — user provides the exact slug.
- **Working (writable, persisted):** `/kaggle/working/` — outputs survive commit, ~20 GB cap.
- **Temp (not persisted):** `/kaggle/temp/` — for large throwaway intermediates.
- **Accelerator:** must be set to `GPU T4 x2`; internet must be **On** (ngrok tunnel + pip).
- **Session:** ~12h GPU runtime, ~4 vCPU. Save checkpoints often.

## Cell 1 — Environment check

```python
import sys, torch
print("Python:", sys.version.split()[0], "| PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
for i in range(n_gpu):
    print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)}")
assert n_gpu == 2, "Set the accelerator to 'GPU T4 x2' in Kaggle notebook settings."
torch.backends.cudnn.benchmark = True          # autotune convolutions for fixed shapes
device = torch.device("cuda")
```

## Cell 2 — Imports + CONFIG

```python
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CONFIG = {
    "input_dir":   Path("/kaggle/input/<dataset-slug>"),  # <-- confirmed path
    "run_name":    "run1",
    "batch_size":  512,      # GLOBAL batch, split across the 2 T4s
    "epochs":      30,
    "lr":          1e-3,
    "num_workers": 4,
    "num_classes": 34,       # <-- from data_description.md
    "seed":        42,
}
torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])

OUT = Path("/kaggle/working") / CONFIG["run_name"]
for sub in ("checkpoints", "logs", "metrics", "artifacts"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
```

## Cell 2b — Run logger (mandatory)

Set this up right after CONFIG. Every message goes to `logs/run.log` **and** the console; the file is line-buffered so a timeout/crash still leaves a readable log. Use `log(...)` everywhere you'd otherwise `print(...)` progress.

```python
import logging, sys

LOG_FILE = OUT / "logs" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    force=True,                                   # override Kaggle/IPython's preconfigured root logger
)
log = logging.getLogger("run").info
log(f"run '{CONFIG['run_name']}' start | {torch.cuda.device_count()}× {torch.cuda.get_device_name(0)}")
log("config: " + json.dumps({k: str(v) for k, v in CONFIG.items()}))   # import json in cell 2
```

## Cell 5 — Dataset + DataLoader

```python
class TabularDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                          num_workers=CONFIG["num_workers"], pin_memory=True)
```

## Cell 7 — Model → DataParallel → AMP

```python
model = MyModel(...).to(device)                 # <-- architecture from the spec
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)              # replicates model, splits each batch over both T4s
scaler    = torch.amp.GradScaler("cuda")        # mixed precision — big T4 speedup
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
criterion = nn.CrossEntropyLoss()               # swap for the task's loss
```

## Cell 8 — Training loop (AMP + best checkpoint)

Log each epoch via `log(...)` and wrap the whole loop in `try/except` so a crash/timeout still writes a traceback to `run.log`.

```python
import traceback
def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

best_val = float("inf")
try:
    for epoch in range(CONFIG["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):    # fp16 forward/backward on the T4 tensor cores
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    val_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss /= len(val_loader.dataset)
        log(f"epoch {epoch:03d}  val_loss {val_loss:.4f}")   # -> console + logs/run.log

        torch.save(unwrap(model).state_dict(), OUT / "checkpoints" / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(unwrap(model).state_dict(), OUT / "checkpoints" / "best.pt")
            log(f"  new best_val {best_val:.4f} -> best.pt")
    log(f"training done | best_val {best_val:.4f}")
except Exception:
    log("run failed:\n" + traceback.format_exc())     # traceback lands in run.log
    raise
```

Save `unwrap(model).state_dict()` — the raw model, not the `DataParallel` wrapper — so the checkpoint reloads cleanly anywhere.

## Cell 8 — track history (mandatory for the curves)

Keep a `history` list during training so cell 11 can plot. Append one dict per epoch/round:

```python
history = []                                    # define before the loop
# ... inside the loop, after computing train/val loss + any tracked metric:
history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_acc": val_acc})            # add whatever you track
```

## Cell 9 — Evaluation (full default metric set)

Classification — always report accuracy + precision/recall/F1 (macro **and** weighted). Reusable helper:

```python
import json
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

def scores(y_true, y_pred):
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "macro_precision":    float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":       float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

metrics = scores(y_true, y_pred)
log("final metrics: " + json.dumps(metrics))       # also captured in logs/run.log
LABELS = list(range(CONFIG["num_classes"]))
rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
(OUT / "metrics" / "summary.json").write_text(json.dumps({"metrics": metrics, "history": history}, indent=2, default=float))
(OUT / "metrics" / "classification_report.json").write_text(json.dumps(rep, indent=2, default=float))
np.save(OUT / "metrics" / "confusion_matrix.npy", confusion_matrix(y_true, y_pred, labels=LABELS))
log(f"metrics + reports saved to {OUT}")
```

Regression instead: report **MAE, RMSE, R²** (e.g. `sklearn.metrics.mean_absolute_error`, `mean_squared_error(squared=False)`, `r2_score`) and skip the confusion matrix.

## Cell 11 — Diagnostic plots (default)

Always produce these; save PNG **and** `plt.show()`. `matplotlib` is preinstalled on Kaggle.

```python
import matplotlib.pyplot as plt

ep = [h["epoch"] for h in history]

# (1) loss curve
plt.figure(figsize=(7, 4.2))
plt.plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
if "val_loss" in history[0]:
    plt.plot(ep, [h["val_loss"] for h in history], marker="s", label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curve")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(OUT / "artifacts" / "loss_curve.png", dpi=120); plt.show()

# (2) metric curve (accuracy or your tracked val metric)
if "val_acc" in history[0]:
    plt.figure(figsize=(7, 4.2))
    plt.plot(ep, [h["val_acc"] for h in history], "b-o")
    plt.xlabel("epoch"); plt.ylabel("val accuracy"); plt.title("Accuracy curve")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT / "artifacts" / "metric_curve.png", dpi=120); plt.show()

# (3) confusion-matrix heatmap (row-normalized)
cm = confusion_matrix(y_true, y_pred, labels=LABELS).astype(float)
cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
plt.figure(figsize=(11, 9.5))
plt.imshow(cmn, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion matrix (row-normalized)")
plt.tight_layout()
plt.savefig(OUT / "artifacts" / "confusion_matrix.png", dpi=120); plt.show()

# (4) per-group comparison bar chart — when the run has natural groups (clients / model families / classes)
# groups = {"g1": {"accuracy":..,"macro_f1":..,"macro_recall":..}, ...}
# x = np.arange(len(groups)); w = 0.26; keys = list(groups)
# for off, m in zip((-w, 0, w), ("accuracy","macro_f1","macro_recall")):
#     plt.bar(x+off, [groups[k][m] for k in keys], w, label=m)
# plt.xticks(x, keys); plt.legend(); plt.savefig(OUT/"artifacts"/"per_group_metrics.png"); plt.show()
```

## Multi-GPU notes

- **`nn.DataParallel`** is the default: one process, replicates the model each step, scatters the batch to both GPUs, gathers the outputs on GPU 0. Simple and notebook-friendly. Keep `batch_size` comfortably larger than 2 so each GPU gets a real slice; GPU 0 carries slightly more load.
- **AMP** (`autocast` + `GradScaler`) is where most of the T4 speedup comes from — do not drop it.
- **BatchNorm** under DataParallel normalizes per-GPU (half-batch stats). For small batches, prefer GroupNorm/LayerNorm.
- **DDP** is faster but awkward in a notebook (needs `spawn`/multiprocessing). Only reach for it if the user explicitly asks; then move training into a script cell launched with `torch.multiprocessing.spawn`.
