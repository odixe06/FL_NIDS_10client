---
description: Build or fix a task10 two-T4 federated benchmark notebook under the locked contract
argument-hint: [method and/or scenario, e.g. "perfed_skd cnn1d" or "regenerate all three"]
---

Use the `kaggle-training-notebook` skill to work on: $ARGUMENTS

Follow its gates in order and do not skip ahead:

1. Read `data_description.md`, the paper `.md`, this folder's
   `ARCHITECTURE_AND_OUTPUT_SPEC.md`, and the shared
   `../ARCHITECTURE_AND_OUTPUT_SPEC.md` in full.
2. Lock the specification and ask me — in **one** consolidated round — about
   anything the docs do not pin down. Never substitute a plausible default.
3. Decide the evaluation scope from state semantics and record it in `CONFIG`.
4. Change the builder or runtime module (notebooks here are generated, not
   hand-edited) and re-run the builder.
5. Run the structural validator and report the result as **structural** unless
   the notebook actually completed on a Kaggle `GPU T4 x2` session.

Keep the locked contract exact: ten clients, client-parallel two-worker
execution on exactly two T4s, per-round checkpoints with no `best.pt`, per-round
evaluation on the complete `global_test_data.csv`, and exactly the ten
comparison metrics.
