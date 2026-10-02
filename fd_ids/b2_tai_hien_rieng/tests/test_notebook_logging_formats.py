#!/usr/bin/env python3
"""Regression checks for stdlib logging format strings in generated notebooks."""

from __future__ import annotations

import ast
import io
import json
import logging
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_GLOB = "*_10_clients/*.ipynb"
GENERATOR_PATH = ROOT / "tools" / "build_fd_ids_notebooks.py"
DDP_RUNTIME_PATH = ROOT / "tools" / "fd_ids_ddp_runtime.py"
LOGGER_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
UNSUPPORTED_COMMA_GROUPING = re.compile(r"%(?:\([^)]+\))?[-+ #0]*,")
PERCENT_TOKEN = re.compile(
    r"%(?:\([^)]+\))?[-+ #0]*(?:\d+)?(?:\.\d+)?[diouxXeEfFgGcrsa%]"
)


def logging_format_strings_from_source(source: str, filename: str):
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "logger"
            and function.attr in LOGGER_METHODS
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.lineno, node.args[0].value, len(node.args) - 1


def logging_format_strings(notebook_path: Path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for line_number, format_string, argument_count in (
            logging_format_strings_from_source(
                source,
                f"{notebook_path}:cell_{cell_index}",
            )
        ):
            yield cell_index, line_number, format_string, argument_count


def representative_logging_args(format_string: str):
    values = []
    for token in PERCENT_TOKEN.findall(format_string):
        conversion = token[-1]
        if conversion == "%":
            continue
        if conversion in "diouxXc":
            values.append(1)
        elif conversion in "eEfFgG":
            values.append(1.25)
        else:
            values.append("value")
    return tuple(values)


class NotebookLoggingFormatTests(unittest.TestCase):
    def test_generator_does_not_emit_unsupported_comma_grouping(self):
        for source_path in (GENERATOR_PATH, DDP_RUNTIME_PATH):
            source = source_path.read_text(encoding="utf-8")
            with self.subTest(source=source_path.name):
                self.assertIsNone(
                    UNSUPPORTED_COMMA_GROUPING.search(source),
                    "Training sources must not emit `%,d` logging formats.",
                )

    def test_lazy_logging_does_not_use_python_comma_grouping(self):
        failures = []
        notebooks = sorted(ROOT.glob(NOTEBOOK_GLOB))
        self.assertEqual(3, len(notebooks))
        for notebook_path in notebooks:
            for (
                cell_index,
                line_number,
                format_string,
                _,
            ) in logging_format_strings(
                notebook_path
            ):
                if UNSUPPORTED_COMMA_GROUPING.search(format_string):
                    failures.append(
                        (
                            str(notebook_path.relative_to(ROOT)),
                            cell_index,
                            line_number,
                            format_string,
                        )
                    )
        self.assertEqual(
            [],
            failures,
            "stdlib logging uses %-interpolation and rejects `%,d`; "
            "pre-format grouped numbers and pass them through `%s`",
        )

    def test_all_lazy_logging_formats_are_runtime_valid(self):
        failures = []
        for notebook_path in sorted(ROOT.glob(NOTEBOOK_GLOB)):
            for (
                cell_index,
                line_number,
                format_string,
                actual_argument_count,
            ) in logging_format_strings(notebook_path):
                representative_args = representative_logging_args(format_string)
                if len(representative_args) != actual_argument_count:
                    failures.append(
                        (
                            str(notebook_path.relative_to(ROOT)),
                            cell_index,
                            line_number,
                            "argument_count",
                            format_string,
                            actual_argument_count,
                            len(representative_args),
                        )
                    )
                    continue
                try:
                    format_string % representative_args
                except (TypeError, ValueError) as error:
                    failures.append(
                        (
                            str(notebook_path.relative_to(ROOT)),
                            cell_index,
                            line_number,
                            type(error).__name__,
                            format_string,
                            str(error),
                        )
                    )
        self.assertEqual([], failures)

    def test_ddp_runtime_lazy_logging_formats_are_runtime_valid(self):
        failures = []
        source = DDP_RUNTIME_PATH.read_text(encoding="utf-8")
        for line_number, format_string, actual_argument_count in (
            logging_format_strings_from_source(source, str(DDP_RUNTIME_PATH))
        ):
            if UNSUPPORTED_COMMA_GROUPING.search(format_string):
                failures.append(
                    (line_number, "unsupported_comma_grouping", format_string)
                )
                continue
            representative_args = representative_logging_args(format_string)
            if len(representative_args) != actual_argument_count:
                failures.append(
                    (
                        line_number,
                        "argument_count",
                        format_string,
                        actual_argument_count,
                        len(representative_args),
                    )
                )
                continue
            try:
                format_string % representative_args
            except (TypeError, ValueError) as error:
                failures.append(
                    (
                        line_number,
                        type(error).__name__,
                        format_string,
                        str(error),
                    )
                )
        self.assertEqual([], failures)

    def test_captured_failure_is_red_capable(self):
        with self.assertRaisesRegex(
            ValueError, r"unsupported format character ','"
        ):
            "Converting %s: %,d rows" % ("client_1_train.csv", 31_520)

    def test_fixed_grouped_integer_log_message(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("fd_ids_logging_regression")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

        logger.info(
            "Converting %s: %s rows",
            "client_1_train.csv",
            f"{31_520:,}",
        )

        self.assertEqual(
            "Converting client_1_train.csv: 31,520 rows",
            stream.getvalue().strip(),
        )


if __name__ == "__main__":
    unittest.main()
