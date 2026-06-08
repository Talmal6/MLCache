from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_h1h0_local_runtime.py"
SPEC = importlib.util.spec_from_file_location("run_h1h0_local_runtime", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def write_runner_npz(path: Path, *, row_count: int = 2) -> None:
    embeddings = np.eye(row_count, dtype=np.float32)
    np.savez(
        path,
        query=np.asarray([f"query {idx}" for idx in range(row_count)], dtype=object),
        anchor=np.asarray([f"anchor {idx}" for idx in range(row_count)], dtype=object),
        h0h1=np.asarray([1 if idx % 2 == 0 else 0 for idx in range(row_count)], dtype=np.int32),
        query_embedding=embeddings,
        anchor_embedding=embeddings,
    )


class H1H0LocalRunnerTests(unittest.TestCase):
    def test_runner_pair_level_mode_writes_summary_and_per_request_csv(self) -> None:
        output = self._run_mode("pair_level")

        self.assertTrue((output / "summary_metrics.json").exists())
        self.assertTrue((output / "per_request_decisions.csv").exists())
        summary = self._read_json(output / "summary_metrics.json")
        rows = self._read_csv(output / "per_request_decisions.csv")
        self.assertEqual(summary["n_requests"], 2)
        self.assertEqual(len(rows), 2)

    def test_runner_compare_scorers_writes_comparison_report(self) -> None:
        output = self._run_mode("pair_level", compare_scorers=True)

        comparison = self._read_json(output / "comparison_report.json")
        cosine_summary = self._read_json(output / "cosine" / "summary_metrics.json")
        ensemble_summary = self._read_json(output / "ensemble" / "summary_metrics.json")

        self.assertEqual(comparison["status"], "compared")
        self.assertEqual(comparison["order"], ["Cosine", "Ensemble"])
        self.assertIn("summary_deltas", comparison)
        self.assertIn("decision_comparison", comparison)
        self.assertEqual(cosine_summary["n_requests"], 2)
        self.assertEqual(ensemble_summary["n_requests"], 2)

    def test_runner_schema_only_writes_schema_without_runtime_outputs(self) -> None:
        output = self._run_schema_only()

        schema = self._read_json(output / "schema_report.json")
        summary = self._read_json(output / "summary_metrics.json")

        self.assertTrue(schema["schema_only"])
        self.assertEqual(schema["row_count"], 20)
        self.assertEqual(schema["selected_row_count"], 20)
        self.assertFalse((output / "per_request_decisions.csv").exists())
        self.assertEqual(summary["status"], "schema_only")

    def test_runner_max_rows_limits_replay(self) -> None:
        output = self._run_mode("pair_level", row_count=20, max_rows=5, no_file_persistence=True)

        summary = self._read_json(output / "summary_metrics.json")
        rows = self._read_csv(output / "per_request_decisions.csv")

        self.assertEqual(summary["n_requests"], 5)
        self.assertEqual(len(rows), 5)

    def test_runner_indexes_anchor_entries_before_replay(self) -> None:
        output = self._run_mode("pair_level")

        vector_store = self._read_json(output / "local_runtime_state" / "vector_store.json")

        self.assertEqual(len(vector_store["records"]), 2)

    def test_runner_computes_false_accepts_correctly(self) -> None:
        output = self._run_mode("pair_level")

        summary = self._read_json(output / "summary_metrics.json")

        self.assertEqual(summary["n_false_accepts"], 1)
        self.assertAlmostEqual(summary["stream_error_rate"], 0.5)

    def test_runner_computes_true_accepts_correctly(self) -> None:
        output = self._run_mode("pair_level")

        summary = self._read_json(output / "summary_metrics.json")

        self.assertEqual(summary["n_true_accepts"], 1)
        self.assertAlmostEqual(summary["h1_recall"], 1.0)

    def test_runner_shadow_mode_writes_query_calibration_records(self) -> None:
        output = self._run_mode("shadow_query_level")

        query_records = self._read_json(output / "local_runtime_state" / "query_calibration_records.json")
        calibration_report = self._read_json(output / "query_level_calibration_report.json")

        self.assertEqual(len(query_records["records"]), 2)
        self.assertEqual(calibration_report["record_count"], 2)

    def test_pair_level_smoke_with_20_rows_completes(self) -> None:
        output = self._run_mode("pair_level", row_count=20, no_file_persistence=True)

        summary = self._read_json(output / "summary_metrics.json")

        self.assertEqual(summary["n_requests"], 20)
        self.assertEqual(summary["n_true_accepts"], 10)
        self.assertEqual(summary["n_false_accepts"], 10)

    def _run_mode(
        self,
        mode: str,
        *,
        row_count: int = 2,
        max_rows: int | None = None,
        no_file_persistence: bool = False,
        compare_scorers: bool = False,
    ) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        npz = root / "pairs.npz"
        output = root / "out"
        write_runner_npz(npz, row_count=row_count)

        command = [
            "--npz",
            str(npz),
            "--output-dir",
            str(output),
            "--top-k",
            "2",
            "--pair-threshold",
            "0.0",
            "--mode",
            mode,
        ]
        if max_rows is not None:
            command.extend(["--max-rows", str(max_rows)])
        if no_file_persistence:
            command.append("--no-file-persistence")
        if compare_scorers:
            command.append("--compare-scorers")

        args = runner.parse_args(command)
        runner.run_experiment(args)
        return output

    def _run_schema_only(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        npz = root / "pairs.npz"
        output = root / "out"
        write_runner_npz(npz, row_count=20)

        args = runner.parse_args(
            [
                "--npz",
                str(npz),
                "--output-dir",
                str(output),
                "--schema-only",
            ]
        )
        runner.run_experiment(args)
        return output

    @staticmethod
    def _read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_csv(path: Path):
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
