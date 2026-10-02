#!/usr/bin/env python3
"""Regression check for NumPy scalar leakage into FedCAPS JSON outputs."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "fedcaps_client_parallel_runtime.py"


class FakeInt64:
    """A JSON-incompatible scalar with NumPy int64's relevant behaviour."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def tolist(self) -> int:
        return self.value


class FakeArray:
    def __init__(self, values) -> None:
        self.values = list(values)

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return FakeArray([self.values[index] for index in indices])
        return self.values[indices]

    def __iter__(self):
        return iter(self.values)

    def astype(self, _dtype):
        return FakeArray([FakeInt64(int(value)) for value in self.values])

    def tolist(self):
        return [int(value) for value in self.values]


class FakeRng:
    def permutation(self, length: int):
        return list(reversed(range(length)))


class FakeNumpy:
    generic = FakeInt64
    ndarray = FakeArray

    def __init__(self) -> None:
        self.random = self

    @staticmethod
    def asarray(values):
        return FakeArray(values)

    @staticmethod
    def default_rng(_seed):
        return FakeRng()


def load_serialization_seam() -> dict:
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"), filename=str(RUNTIME))
    wanted_functions = {"seed_from", "json_compatible", "atomic_json", "augment_records", "records_for_csv"}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FEATURE_COLUMNS" for target in node.targets):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    class FakeTorch:
        class Tensor:
            pass

    namespace = {"hashlib": hashlib, "json": json, "np": FakeNumpy(), "torch": FakeTorch, "Path": Path}
    exec(compile(ast.fix_missing_locations(module), str(RUNTIME), "exec"), namespace)
    return namespace


def main() -> None:
    seam = load_serialization_seam()
    source = [{
        "record_id": "c01_r001",
        "client_id": 1,
        "collection_epoch": 1,
        "feature_ids": [0, 3, 7, 20],
        "feature_names": ["a", "b", "c", "d"],
        "performance_micro_f1": 0.75,
    }]
    augmented = seam["augment_records"](source, 25, 42)
    feature_values = [value for record in augmented for value in record["feature_ids"]]
    assert all(type(value) is int for value in feature_values), {
        "observed_types": sorted({type(value).__name__ for value in feature_values})
    }
    json.dumps(augmented)
    json.dumps(seam["records_for_csv"](augmented))
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "record.json"
        seam["atomic_json"](output, {"feature_id": FakeInt64(7)})
        assert json.loads(output.read_text(encoding="utf-8")) == {"feature_id": 7}
    print("[OK] FedCAPS augmented records are JSON serializable with built-in int feature IDs")


if __name__ == "__main__":
    main()
