from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlcache.feedback import (
    H1H0NPZDataset,
    H1H0NPZJudgeAdapter,
    H1H0NPZStreamAdapter,
    JudgeLabel,
    JudgeRequest,
)
from mlcache.semantic_types import CacheKey, Query


def write_default_npz(path: Path, **overrides: object) -> None:
    data = {
        "query": np.asarray(["query one", "query zero"], dtype=object),
        "anchor": np.asarray(["anchor one", "anchor zero"], dtype=object),
        "h0h1": np.asarray([1, 0], dtype=np.int32),
        "query_embedding": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "anchor_embedding": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    data.update(overrides)
    np.savez(path, **data)


class H1H0NPZAdapterTests(unittest.TestCase):
    def test_dataset_adapter_detects_fields_from_synthetic_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            write_default_npz(path)

            dataset = H1H0NPZDataset(path)

            self.assertEqual(dataset.schema.query_field, "query")
            self.assertEqual(dataset.schema.anchor_field, "anchor")
            self.assertEqual(dataset.schema.label_field, "h0h1")
            self.assertEqual(dataset.schema.query_embedding_field, "query_embedding")
            self.assertEqual(dataset.schema.anchor_embedding_field, "anchor_embedding")
            self.assertEqual(dataset.schema.row_count, 2)

    def test_dataset_adapter_respects_explicit_field_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            np.savez(
                path,
                request=np.asarray(["q"], dtype=object),
                candidate=np.asarray(["a"], dtype=object),
                h0h1=np.asarray([1], dtype=np.int32),
                x=np.asarray([[1.0, 0.0]], dtype=np.float32),
                anchor_vec=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

            dataset = H1H0NPZDataset(
                path,
                query_field="request",
                anchor_field="candidate",
                query_embedding_field="x",
                anchor_embedding_field="anchor_vec",
            )

            self.assertEqual(dataset.schema.query_field, "request")
            self.assertEqual(dataset.schema.anchor_field, "candidate")
            self.assertEqual(dataset.schema.query_embedding_field, "x")
            self.assertEqual(dataset.schema.anchor_embedding_field, "anchor_vec")

    def test_dataset_adapter_reuses_query_and_embedding_when_anchor_fields_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            np.savez(
                path,
                text=np.asarray(["q1", "q2"], dtype=object),
                label=np.asarray([1, 0], dtype=np.int32),
                emb=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )

            dataset = H1H0NPZDataset(path)

            self.assertEqual(dataset.schema.query_field, "text")
            self.assertEqual(dataset.schema.anchor_field, "text")
            self.assertEqual(dataset.schema.query_embedding_field, "emb")
            self.assertEqual(dataset.schema.anchor_embedding_field, "emb")

    def test_dataset_adapter_falls_back_to_label_field_when_h0h1_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            write_default_npz(path)
            with np.load(path, allow_pickle=True) as loaded:
                data = {key: loaded[key] for key in loaded.files if key != "h0h1"}
            data["label"] = np.asarray([1, 0], dtype=np.int32)
            np.savez(path, **data)

            dataset = H1H0NPZDataset(path)

            self.assertEqual(dataset.schema.label_field, "label")

    def test_global_cluster_field_is_used_as_anchor_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            np.savez(
                path,
                text=np.asarray(["q1", "q2"], dtype=object),
                global_cluster=np.asarray([42, 42], dtype=np.int32),
                label=np.asarray([1, 0], dtype=np.int32),
                emb=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )

            dataset = H1H0NPZDataset(path)
            stream = H1H0NPZStreamAdapter(dataset)
            entries = stream.anchor_entries()

            self.assertEqual(dataset.schema.anchor_field, "global_cluster")
            self.assertEqual(dataset.schema.label_field, "label")
            self.assertEqual(dataset.records()[0].anchor_key, CacheKey("anchor:42"))
            self.assertEqual(len(entries), 1)
            self.assertEqual(tuple(entries[0].embedding), (0.5, 0.5))

    def test_global_cluster_centroids_use_selected_max_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            np.savez(
                path,
                text=np.asarray(["q1", "q2", "q3"], dtype=object),
                global_cluster=np.asarray([7, 7, 8], dtype=np.int32),
                label=np.asarray([1, 0, 1], dtype=np.int32),
                emb=np.asarray([[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]], dtype=np.float32),
            )

            dataset = H1H0NPZDataset(path, max_rows=2)
            stream = H1H0NPZStreamAdapter(dataset)
            entries = stream.anchor_entries()

            self.assertEqual(len(dataset.records()), 2)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].cache_key, CacheKey("anchor:7"))
            self.assertEqual(tuple(entries[0].embedding), (0.5, 0.5))

    def test_dataset_adapter_rejects_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            write_default_npz(path, h0h1=np.asarray([1, 2], dtype=np.int32))

            with self.assertRaisesRegex(ValueError, "invalid labels"):
                H1H0NPZDataset(path)

    def test_stream_adapter_produces_lookup_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            write_default_npz(path)
            stream = H1H0NPZStreamAdapter(H1H0NPZDataset(path))

            record, lookup = next(stream.iter_records_and_lookups())

            self.assertEqual(lookup.query, record.query)
            self.assertEqual(tuple(lookup.embedding), record.query_embedding)
            self.assertEqual(lookup.metadata.attributes["query_id"], record.query_id)
            self.assertEqual(lookup.metadata.attributes["row_id"], record.row_id)
            self.assertEqual(lookup.metadata.attributes["expected_anchor_key"], str(record.anchor_key))
            self.assertEqual(lookup.metadata.attributes["h0h1"], record.label)
            self.assertEqual(lookup.metadata.attributes["source"], "h1h0_npz")

    def test_stream_adapter_deduplicates_anchor_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            write_default_npz(
                path,
                anchor=np.asarray(["same anchor", "same anchor"], dtype=object),
                anchor_embedding=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            )
            stream = H1H0NPZStreamAdapter(H1H0NPZDataset(path))

            entries = stream.anchor_entries()

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].metadata.attributes["source"], "h1h0_npz_anchor")
            self.assertEqual(entries[0].metadata.attributes["anchor_observation_count"], 2)

    def test_judge_returns_reusable_for_label_one(self) -> None:
        record, judge = self._first_record_and_judge()

        result = judge.judge(
            JudgeRequest(
                query=record.query,
                candidate_key=record.anchor_key,
                context={"query_id": record.query_id},
            )
        )

        self.assertEqual(result.decision.label, JudgeLabel.REUSABLE)
        self.assertEqual(result.decision.rationale, "h1h0_label_reusable")

    def test_judge_returns_not_reusable_for_label_zero(self) -> None:
        records, judge = self._records_and_judge()
        record = records[1]

        result = judge.judge(
            JudgeRequest(
                query=record.query,
                candidate_key=record.anchor_key,
                context={"query_id": record.query_id},
            )
        )

        self.assertEqual(result.decision.label, JudgeLabel.NOT_REUSABLE)
        self.assertEqual(result.decision.rationale, "h1h0_label_not_reusable")

    def test_judge_returns_uncertain_for_label_minus_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.npz"
            write_default_npz(path, h0h1=np.asarray([-1, 1], dtype=np.int32))
            dataset = H1H0NPZDataset(path)
            records = dataset.records()
            judge = H1H0NPZJudgeAdapter(records)

            result = judge.judge(
                JudgeRequest(
                    query=records[0].query,
                    candidate_key=records[0].anchor_key,
                    context={"query_id": records[0].query_id},
                )
            )

            self.assertEqual(result.decision.label, JudgeLabel.UNCERTAIN)
            self.assertEqual(result.decision.rationale, "h1h0_label_unknown")

    def test_judge_returns_uncertain_for_unknown_pair(self) -> None:
        record, judge = self._first_record_and_judge()

        result = judge.judge(
            JudgeRequest(
                query=Query("unknown query"),
                candidate_key=CacheKey("anchor:missing"),
                candidate_query=Query("unknown anchor"),
            )
        )

        self.assertEqual(result.decision.label, JudgeLabel.UNCERTAIN)
        self.assertEqual(result.decision.rationale, "h1h0_pair_not_found")

    @staticmethod
    def _records_and_judge():
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "pairs.npz"
        write_default_npz(path)
        dataset = H1H0NPZDataset(path)
        records = dataset.records()
        judge = H1H0NPZJudgeAdapter(records)
        dataset.close()
        tmp.cleanup()
        return records, judge

    @classmethod
    def _first_record_and_judge(cls):
        records, judge = cls._records_and_judge()
        return records[0], judge


if __name__ == "__main__":
    unittest.main()
