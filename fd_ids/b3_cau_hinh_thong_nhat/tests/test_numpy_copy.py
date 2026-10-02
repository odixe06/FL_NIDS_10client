from __future__ import annotations

import ast
import warnings
import unittest
from pathlib import Path
from types import SimpleNamespace


RUNTIME_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "common_client_parallel_runtime.py"
)


class FakeArray:
    def __init__(self, rows: int, writable: bool) -> None:
        self.rows = rows
        self.shape = (rows, 1)
        self.dtype = "float32"
        self.writable = writable

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.rows)
            return FakeArray(len(range(start, stop, step)), self.writable)
        return self


class FakeTarget:
    def __getitem__(self, _index):
        return self

    def copy_(self, _source, non_blocking: bool):
        return self


class FakeNumpy:
    ndarray = FakeArray

    @staticmethod
    def empty(_shape, dtype):
        return FakeArray(1, writable=True)

    @staticmethod
    def asarray(array):
        return array

    @staticmethod
    def array(array, copy: bool):
        return FakeArray(len(array), writable=copy)


class FakeTorch:
    device = object

    @staticmethod
    def empty(_shape, dtype, device):
        return FakeTarget()

    @staticmethod
    def from_numpy(array):
        if not array.writable:
            warnings.warn(
                "The given NumPy array is not writable",
                UserWarning,
                stacklevel=2,
            )
        return SimpleNamespace(dtype="float32")


def load_copy_numpy_to_gpu():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(RUNTIME_PATH))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "copy_numpy_to_gpu"
    )
    namespace = {"np": FakeNumpy, "torch": FakeTorch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNTIME_PATH), "exec"), namespace)
    return namespace["copy_numpy_to_gpu"]


class NumpyCopyTest(unittest.TestCase):
    def test_read_only_memmap_chunk_is_copied_before_torch_conversion(self) -> None:
        copy_numpy_to_gpu = load_copy_numpy_to_gpu()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copy_numpy_to_gpu(FakeArray(4, writable=False), "cuda:0", 2)

        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
