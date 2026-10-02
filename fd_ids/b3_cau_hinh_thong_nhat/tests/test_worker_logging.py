from __future__ import annotations

import ast
import hashlib
import json
import logging
import queue
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace


RUNTIME_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "common_client_parallel_runtime.py"
)


def load_worker_logging_seam() -> SimpleNamespace:
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(RUNTIME_PATH))
    required_functions = {
        "configure_logger",
        "seed_from",
        "seed_everything",
        "worker_main",
    }
    selected_nodes = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in required_functions
    ]
    seam_module = ast.Module(body=selected_nodes, type_ignores=[])
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda _seed: None,
            set_device=lambda _gpu_id: None,
        ),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False, benchmark=False),
        ),
        manual_seed=lambda _seed: None,
        device=lambda *_args: "fake_cuda_device",
    )
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "logging": logging,
        "np": SimpleNamespace(random=SimpleNamespace(seed=lambda _seed: None)),
        "sys": sys,
        "torch": fake_torch,
        "traceback": traceback,
    }
    exec(compile(seam_module, str(RUNTIME_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


RUNTIME = load_worker_logging_seam()


class WorkerLoggingTest(unittest.TestCase):
    def test_successful_worker_writes_a_nonempty_diagnostic_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "output_dir": str(root / "output"),
                            "seed": 42,
                        }
                    }
                ),
                encoding="utf-8",
            )
            task_queue = queue.Queue()
            result_queue = queue.Queue()
            task_queue.put({"task_id": "shutdown_gpu_0", "kind": "shutdown"})

            RUNTIME.worker_main(0, task_queue, result_queue, str(manifest_path))

            self.assertEqual(result_queue.get_nowait()["kind"], "shutdown_ok")
            worker_log = root / "output" / "logs" / "worker_0.log"
            self.assertTrue(worker_log.is_file())
            self.assertGreater(worker_log.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
