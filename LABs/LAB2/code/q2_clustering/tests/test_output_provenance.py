"""Final-output checks must reject missing or contradictory source records."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from utils.results import export_selected, save_candidate
import extras
from utils.data import validate_config
from utils.results import save_json, sha256


class OutputProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.input = self.root / "features.npz"
        np.savez(self.input, X=np.array([[-3.], [-1.], [1.], [3.]], dtype=np.float32))
        self.key = "kmeans_k2_seed17"
        self.candidate = self.output / "candidates" / self.key
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        row = dict(key=self.key, method="kmeans", k=2, seed=17, status="succeeded",
                   best_restart=0, objective=4., objective_name="inertia", objective_space="raw")
        fit = dict(labels=labels, centers=np.array([[-2.], [2.]], dtype=np.float32),
                   restarts=[dict(restart=0, objective=4.)])
        save_candidate(self.output, row, fit, np.zeros(1), np.ones(1))
        # Evaluation adds fields after fit.json has already been saved.
        row["silhouette"] = .5
        export_selected(self.output, row, labels, development=True)
        selection = dict(status="selected", rule=dict(metric="silhouette"), method="kmeans",
                         k=2, run_key=self.key, restart=0)
        save_json(self.output / "selection.json", selection)
        save_json(self.output / "runs.json", [row])
        cfg = json.loads((PACKAGE / "config.json").read_text())
        cfg.update(candidate_k=[2, 3], silhouette_sample_size=4)
        cfg = validate_config(cfg, development=True)
        save_json(self.output / "config.json", cfg)
        save_json(self.output / "fit_config.json", cfg)
        plan = dict(schema_version=3, input_sha256=sha256(self.input), development=True,
                    config_sha256=sha256(self.output / "config.json"),
                    fit_config_sha256=sha256(self.output / "fit_config.json"))
        save_json(self.output / "plan.json", plan)
        save_json(self.output / "results.json", dict(
            plan=plan, config=cfg,
            selected=row, selection=selection, runs=[row], clusters_sha256=sha256(self.output / "clusters.npy")))

    def check(self):
        args = argparse.Namespace(input=self.input, output=self.output, development=True)
        return extras.check_output(args)

    def change(self, name, change):
        path = self.output / name
        original = path.read_bytes()
        value = json.loads(original)
        change(value)
        save_json(path, value)
        return original

    def test_valid_export_passes(self):
        result = self.check()
        self.assertTrue(result["passed"])
        self.assertNotIn("row_mismatches", result)
        self.assertNotIn("audited_objective", result)

    def test_changed_saved_config_or_fit_config_fails_check(self):
        original = self.change("config.json", lambda data: data.update(standardize=True))
        try:
            with self.assertRaisesRegex(ValueError, "config.json.*hash mismatch"):
                self.check()
        finally:
            (self.output / "config.json").write_bytes(original)
        self.change("fit_config.json", lambda data: data.update(standardize=True))
        with self.assertRaisesRegex(ValueError, "fit_config.json.*hash mismatch"):
            self.check()

    def test_embedded_config_and_plan_must_match_saved_files(self):
        for field in ("config", "plan"):
            with self.subTest(field=field):
                original = self.change("results.json", lambda data: data[field].update(extra="changed"))
                try:
                    with self.assertRaisesRegex(ValueError, field + ".json"):
                        self.check()
                finally:
                    (self.output / "results.json").write_bytes(original)

    def test_selection_must_match_result(self):
        for field, value in (("run_key", "does_not_exist"), ("status", "not_requested"),
                             ("method", "gmm"), ("k", 3), ("restart", 1)):
            with self.subTest(field=field):
                original = self.change("selection.json", lambda data: data.update({field: value}))
                try:
                    with self.assertRaisesRegex(ValueError, "selection"):
                        self.check()
                finally:
                    (self.output / "selection.json").write_bytes(original)

    def test_matching_selection_copies_must_still_identify_selected_run(self):
        self.change("selection.json", lambda data: data.update(run_key="does_not_exist"))
        self.change("results.json", lambda data: data["selection"].update(run_key="does_not_exist"))
        with self.assertRaisesRegex(ValueError, "selection"):
            self.check()

    def test_required_source_files_cannot_be_omitted(self):
        paths = ["selection.json", "export_manifest.json", "runs.json",
                 f"candidates/{self.key}/fit.json", f"candidates/{self.key}/model.npz",
                 f"candidates/{self.key}/clusters.npy"]
        for name in paths:
            with self.subTest(name=name):
                path = self.output / name
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaisesRegex(ValueError, "missing required"):
                        self.check()
                finally:
                    path.write_bytes(original)

    def test_missing_manifest_and_candidate_directory_fails(self):
        (self.output / "export_manifest.json").unlink()
        shutil.rmtree(self.output / "candidates")
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.check()

    def test_export_identity_and_hashes_must_match(self):
        for field, value in (("run_key", "does_not_exist"), ("seed", 42), ("restart", 1),
                             ("model_sha256", "0" * 64), ("clusters_sha256", "0" * 64)):
            with self.subTest(field=field):
                original = self.change("export_manifest.json", lambda data: data.update({field: value}))
                try:
                    with self.assertRaisesRegex(ValueError, "export"):
                        self.check()
                finally:
                    (self.output / "export_manifest.json").write_bytes(original)

    def test_selected_run_and_fit_record_must_match(self):
        changes = [
            ("runs.json", lambda data: data[0].update(seed=42)),
            ("results.json", lambda data: data["selected"].update(objective=999.)),
            (f"candidates/{self.key}/fit.json", lambda data: data["run"].update(seed=42)),
            ("runs.json", lambda data: data.append(dict(data[0]))),
        ]
        for name, change in changes:
            with self.subTest(name=name):
                original = self.change(name, change)
                try:
                    with self.assertRaisesRegex(ValueError, "run|fit"):
                        self.check()
                finally:
                    (self.output / name).write_bytes(original)

    def test_changed_candidate_files_fail(self):
        for name in ("model.npz", "clusters.npy"):
            with self.subTest(name=name):
                path = self.candidate / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaisesRegex(ValueError, "candidate"):
                        self.check()
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
