#!/usr/bin/env python3
"""Regression checks for the shared FD-IDS training configuration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("*_10_clients/*.ipynb"))


class TrainingContractTests(unittest.TestCase):
    def test_all_notebooks_use_the_locked_large_batch_configuration(self):
        self.assertEqual(3, len(NOTEBOOKS))
        for notebook_path in NOTEBOOKS:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
            )
            with self.subTest(notebook=notebook_path.name):
                self.assertIn('"batch_size": 1024', source)
                self.assertIn('"batch_size_per_gpu": 512', source)
                self.assertIn('"gradient_accumulation_steps": 1', source)
                self.assertIn(
                    '"learning_rate_scaling": "none_adam_preserve_paper_lr"',
                    source,
                )
                self.assertIn('"learning_rate": 0.001', source)
                self.assertIn('"num_workers_per_process": 2', source)
                self.assertIn('"num_workers_total": 4', source)
                self.assertIn('"local_epochs": 1', source)
                self.assertIn('"communication_rounds": 10', source)
                self.assertIn('"world_size": 2', source)
                self.assertIn('"distributed_backend": "nccl"', source)
                self.assertIn(
                    '"multi_gpu": "DistributedDataParallel"',
                    source,
                )
                self.assertIn('"torch.distributed.run"', source)
                self.assertIn('"--nproc-per-node=', source)
                self.assertNotIn('"batch_size": 128', source)
                self.assertNotIn("nn.DataParallel", source)

    def test_architecture_spec_documents_batch_semantics(self):
        specification = (
            ROOT / "ARCHITECTURE_AND_OUTPUT_SPEC.md"
        ).read_text(encoding="utf-8")
        self.assertIn("| Global batch size | 1024 |", specification)
        self.assertIn("| Nominal batch size mỗi GPU | 512 |", specification)
        self.assertIn("| Gradient accumulation | Không (`1` step) |", specification)
        self.assertIn("| Local epochs | 1 |", specification)
        self.assertIn("DistributedDataParallel", specification)
        self.assertIn("NCCL / `torchrun`", specification)
        self.assertIn("Adam", specification)
        self.assertIn("0,001", specification)


if __name__ == "__main__":
    unittest.main()
