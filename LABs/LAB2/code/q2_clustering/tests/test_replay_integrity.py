"""Replay keeps fitting settings intact without copying or requiring source code."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import extras
from utils.results import save_json, sha256


class Prepared(Exception):
    pass


class ReplayIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "fit"
        self.source.mkdir()
        self.cfg = json.loads((PACKAGE / "config.json").read_text())
        self.cfg.update(candidate_k=[2, 3], silhouette_sample_size=6)
        save_json(self.source / "config.json", self.cfg)
        save_json(self.source / "fit_config.json", self.cfg)
        self.input = self.root / "features.npz"
        np.savez(self.input, X=np.arange(24, dtype=np.float32).reshape(8, 3))
        self.plan = dict(schema_version=3, device="cpu", development=True,
                         input_sha256=sha256(self.input),
                         config_sha256=sha256(self.source / "config.json"),
                         fit_config_sha256=sha256(self.source / "fit_config.json"))
        self.save_plan()
        self.args = argparse.Namespace(stage="evaluate", experiment=self.source, config=None,
                                       device="cpu", dtype=None, input=self.input, development=True,
                                       output=self.root / "replayed", implementation=PACKAGE / "submission.py")

    def save_plan(self):
        save_json(self.source / "plan.json", self.plan)

    def assert_prepared(self, **expected):
        def prepare(path, cfg, **kwargs):
            for name, value in expected.items():
                self.assertEqual(cfg[name], value)
            raise Prepared
        with patch.object(extras, "prepare_data", side_effect=prepare):
            with self.assertRaises(Prepared):
                extras.evaluate_saved(self.args)

    def assert_rejected(self, message):
        with patch.object(extras, "prepare_data") as prepare:
            with self.assertRaisesRegex(ValueError, message):
                extras.evaluate_saved(self.args)
            prepare.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_config_only_source_is_accepted(self):
        self.assertFalse((self.source / "source_snapshot").exists())
        self.assert_prepared(standardize=False, dtype="float32")

    def test_modified_saved_config_is_rejected(self):
        save_json(self.source / "config.json", dict(self.cfg, standardize=True))
        self.assert_rejected("config.json.*hash mismatch")

    def test_external_config_does_not_bypass_modified_source(self):
        self.args.config = self.root / "evaluation.json"
        save_json(self.args.config, self.cfg)
        save_json(self.source / "config.json", dict(self.cfg, standardize=True))
        self.assert_rejected("config.json.*hash mismatch")

    def test_modified_fitting_config_is_rejected(self):
        save_json(self.source / "fit_config.json", dict(self.cfg, standardize=True))
        self.assert_rejected("fit_config.json.*hash mismatch")

    def test_missing_records_or_hashes_are_rejected(self):
        for relative in ("config.json", "fit_config.json"):
            with self.subTest(missing_file=relative):
                path = self.source / relative
                original = path.read_bytes()
                path.unlink()
                try:
                    self.assert_rejected("missing.*saved")
                finally:
                    path.write_bytes(original)
        for field in ("config_sha256", "fit_config_sha256"):
            with self.subTest(missing_hash=field):
                original = self.plan.pop(field)
                self.save_plan()
                try:
                    self.assert_rejected("missing.*hash")
                finally:
                    self.plan[field] = original
                    self.save_plan()

    def test_evaluation_overrides_are_allowed(self):
        changes = dict(selection_metric="calinski_harabasz", silhouette_sample_size=4,
                       silhouette_seed=123, threads=1)
        self.args.config = self.root / "evaluation.json"
        save_json(self.args.config, dict(self.cfg, **changes))
        self.assert_prepared(**changes)

    def test_fitting_overrides_still_require_refitting(self):
        self.args.config = self.root / "evaluation.json"
        for name, value in (("standardize", True), ("dtype", "float64"),
                            ("candidate_k", [3, 4]), ("seeds", [1, 2]), ("n_init", 9)):
            with self.subTest(field=name):
                save_json(self.args.config, dict(self.cfg, **{name: value}))
                self.assert_rejected("cannot change fitting configuration")

    def test_replay_requires_the_shared_implementation_file(self):
        self.args.implementation = self.root / "not_present.py"
        with self.assertRaisesRegex(FileNotFoundError, "not_present.py"):
            extras.evaluate_saved(self.args)
        self.assertFalse(self.args.output.exists())

    def test_replay_loads_shared_implementation_once_without_fitting(self):
        self.args.implementation = self.root / "custom_submission.py"
        self.args.implementation.write_text("# Custom algorithm and evaluation hooks\n")
        implementation = object()
        with patch.object(extras, "load_implementation", return_value=implementation) as load, \
             patch.object(extras, "copy_saved_runs", return_value=([], {})), \
             patch.object(extras, "evaluate_candidates", side_effect=Prepared) as evaluate, \
             patch("train.fit_candidates") as fit:
            with self.assertRaises(Prepared):
                extras.evaluate_saved(self.args)
        load.assert_called_once_with(self.args.implementation)
        self.assertIs(evaluate.call_args.args[4], implementation)
        fit.assert_not_called()

    def test_legacy_config_location_is_accepted_without_source_files(self):
        legacy = self.source / "source_snapshot"
        legacy.mkdir()
        (self.source / "fit_config.json").rename(legacy / "fit_config.json")
        self.plan.update(schema_version=2, fit_source_hashes={"core.py": "old hash"})
        self.save_plan()
        self.assert_prepared(standardize=False)

    def test_legacy_missing_fit_hash_still_checks_fitting_settings(self):
        self.plan.update(schema_version=2)
        self.plan.pop("fit_config_sha256")
        self.save_plan()
        self.assert_prepared(standardize=False)
        save_json(self.source / "fit_config.json", dict(self.cfg, standardize=True))
        self.assert_rejected("fitting configuration.*fit_config.json")


if __name__ == "__main__":
    unittest.main()
