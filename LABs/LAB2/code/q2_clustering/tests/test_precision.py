"""Precision contracts for preprocessing, metric hooks and replay."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from utils.data import prepare_data
from evaluate import evaluate_partition
import extras
from utils import runtime
from utils.results import sha256


class PrecisionTests(unittest.TestCase):
    def tearDown(self):
        runtime.configure_dtype("float32")

    def test_preprocessing_rounds_only_after_computing_statistics(self):
        X = np.array([[1., 10.], [3., 10.], [9., 10.]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(path, X=X)
            for precision in ("float32", "float64"):
                for standardize in (False, True):
                    with self.subTest(dtype=precision, standardize=standardize):
                        cfg = dict(dtype=precision, standardize=standardize, candidate_k=[2, 3],
                                   silhouette_sample_size=3, silhouette_seed=17)
                        data = prepare_data(path, cfg, development=True)
                        self.assertEqual(data["fit_features"].dtype, np.dtype(precision))
                        self.assertEqual(data["summary"]["compute_dtype"], precision)
                        expected = ((X.astype(np.float64) - data["mean"]) / data["scale"]).astype(precision)
                        np.testing.assert_array_equal(data["fit_features"], expected)
                        self.assertTrue(np.isfinite(data["fit_features"]).all())

    def test_metrics_receive_selected_precision_and_integer_labels(self):
        X = np.arange(12, dtype=np.float64).reshape(6, 2)
        for precision in ("float32", "float64"):
            with self.subTest(dtype=precision):
                runtime.configure_dtype(precision)
                seen = []

                def score(features, labels):
                    self.assertEqual(features.dtype, getattr(torch, precision))
                    self.assertEqual(labels.dtype, torch.int64)
                    seen.append(len(features))
                    return features.new_tensor(.5)

                hooks = SimpleNamespace(**{name: score for name in
                                        ("silhouette_score", "calinski_harabasz_score", "davies_bouldin_score")})
                record = evaluate_partition(X, [0, 0, 0, 1, 1, 1], [0, 1, 3, 4], hooks)
                self.assertEqual(seen, [4, 6, 6])
                self.assertTrue(all(m["status"] == "ok" for m in record["metrics"].values()))

    def test_replay_inherits_precision_and_rejects_precision_changes(self):
        cfg = json.loads((PACKAGE / "config.json").read_text())
        for saved in (None, "float32", "float64"):
            with self.subTest(saved_dtype=saved), tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                original = dict(cfg)
                original.pop("dtype", None)
                if saved:
                    original["dtype"] = saved
                (source / "config.json").write_text(json.dumps(original))
                (source / "fit_config.json").write_text(json.dumps(original))
                input_path = source / "features.npz"
                np.savez(input_path, X=np.ones((6, 2), dtype=np.float32))
                plan = dict(device="cpu", development=True, input_sha256=sha256(input_path),
                            config_sha256=sha256(source / "config.json"),
                            fit_config_sha256=sha256(source / "fit_config.json"))
                (source / "plan.json").write_text(json.dumps(plan))
                expected = saved or "float64"
                args = argparse.Namespace(stage="evaluate", experiment=source, config=None, device="cpu",
                                          dtype=None, input=source / "features.npz", development=True)

                class Prepared(Exception):
                    pass

                def prepare(path, actual, **kwargs):
                    self.assertEqual(actual["dtype"], expected)
                    self.assertEqual(runtime.dtype, getattr(torch, expected))
                    raise Prepared

                with patch.object(extras, "prepare_data", side_effect=prepare):
                    with self.assertRaises(Prepared):
                        extras.evaluate_saved(args)
                    args.dtype = "float64" if expected == "float32" else "float32"
                    with self.assertRaisesRegex(ValueError, "cannot change fitting configuration"):
                        extras.evaluate_saved(args)


if __name__ == "__main__":
    unittest.main()
