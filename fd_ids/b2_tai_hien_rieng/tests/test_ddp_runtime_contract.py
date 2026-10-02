#!/usr/bin/env python3
"""Static contract checks for the two-GPU Kaggle DDP runtime."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "tools" / "fd_ids_ddp_runtime.py"
NOTEBOOKS = sorted(ROOT.glob("*_10_clients/*.ipynb"))


class DdpRuntimeContractTests(unittest.TestCase):
    def test_runtime_is_valid_python_and_uses_one_process_per_gpu(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(RUNTIME_PATH))
        self.assertIn(
            "from torch.nn.parallel import DistributedDataParallel as DDP",
            source,
        )
        self.assertIn("dist.init_process_group(", source)
        self.assertIn('backend=CONFIG["distributed_backend"]', source)
        self.assertIn("torch.cuda.set_device(local_rank)", source)
        self.assertIn("device_ids=[local_rank]", source)
        self.assertNotIn("nn.DataParallel", source)

    def test_training_and_evaluation_have_distinct_distributed_samplers(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("DistributedSampler(", source)
        self.assertIn("class DistributedEvalSampler(Sampler):", source)
        self.assertIn("sampler.set_epoch(", source)
        self.assertIn("assert example_count == len(dataset)", source)
        self.assertIn('"sampler_padding_rows"', source)

    def test_rank_zero_owns_fedavg_and_outputs(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("if rank == 0:\n                add_weighted_state(", source)
        self.assertIn("broadcast_model_state(global_model, source_rank=0)", source)
        self.assertIn('"rank_1_log": "logs/rank_1.log"', source)
        self.assertIn("ddp_gradient_allreduce_bytes_all_ranks", source)
        self.assertIn('"cuda_gpu0_peak_allocated_bytes"', source)
        self.assertIn('"cuda_gpu1_peak_reserved_bytes"', source)
        self.assertIn('"total_memory_bytes"', source)

    def test_generated_notebooks_embed_the_runtime_and_torchrun_launcher(self):
        self.assertEqual(3, len(NOTEBOOKS))
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
        for notebook_path in NOTEBOOKS:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
            )
            embedded_runtime = None
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                for node in tree.body:
                    if (
                        isinstance(node, ast.Assign)
                        and any(
                            isinstance(target, ast.Name)
                            and target.id == "DDP_RUNTIME_SOURCE"
                            for target in node.targets
                        )
                        and isinstance(node.value, ast.Constant)
                    ):
                        embedded_runtime = node.value.value
            with self.subTest(notebook=notebook_path.name):
                self.assertEqual(runtime_source, embedded_runtime)
                self.assertIn('"torch.distributed.run"', source)
                self.assertIn('"--standalone"', source)
                self.assertIn('"--max-restarts=0"', source)
                self.assertIn('f"--nproc-per-node={CONFIG[', source)
                self.assertIn('"FD_IDS_MANIFEST"', source)


if __name__ == "__main__":
    unittest.main()
