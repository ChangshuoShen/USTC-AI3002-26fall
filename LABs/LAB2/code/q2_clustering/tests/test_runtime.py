import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from utils.runtime import resolve_device, validate_device
from utils import runtime
import train
from extras import torch_check_adapter


class RuntimeTests(unittest.TestCase):
    def tearDown(self):
        runtime.configure_device("cpu")
        runtime.configure_dtype("float32")

    def test_fitter_passes_seeded_torch_generator(self):
        class ProbeComplete(Exception):
            pass

        def initialization(features, count, rng):
            self.assertIsInstance(features, torch.Tensor)
            self.assertEqual(features.dtype, torch.float32)
            self.assertIsInstance(rng, torch.Generator)
            self.assertEqual(rng.device, features.device)
            raise ProbeComplete

        with self.assertRaises(ProbeComplete):
            train.fit_kmeans(torch.zeros((4, 2), dtype=torch.float64), 2, 17,
                            SimpleNamespace(init_centers_kmeanspp=initialization))

    def test_cuda_device_has_explicit_index(self):
        with patch.object(runtime, "validate_device"):
            self.assertEqual(runtime.configure_device("cuda"), torch.device("cuda:0"))

    def test_cpu_hooks_receive_tensors(self):
        import numpy as np

        def centers(features, count, rng):
            self.assertIsInstance(features, torch.Tensor)
            self.assertEqual(features.dtype, torch.float32)
            self.assertEqual(features.device, runtime.device)
            return features[:count].clone()

        wrapped = torch_check_adapter(SimpleNamespace(init_centers_kmeanspp=centers))
        np.testing.assert_array_equal(wrapped.init_centers_kmeanspp(np.ones((4, 2)), 2, None),
                                      np.ones((2, 2)))

    def test_hooks_cannot_mutate_public_inputs(self):
        import numpy as np

        def centers(features, count, rng):
            features.add_(1)
            return features[:count]

        wrapped = torch_check_adapter(SimpleNamespace(init_centers_kmeanspp=centers))
        with self.assertRaisesRegex(ValueError, "modified an input"):
            wrapped.init_centers_kmeanspp(np.ones((4, 2)), 2, None)

    def test_numpy_algorithm_output_is_rejected_on_cpu(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, "float32 tensor"):
            train.validate_tensor(np.ones((2, 2)), (2, 2), "student output")

    def test_float64_is_optional_for_fit_and_public_checks(self):
        import numpy as np

        runtime.configure_dtype("float64")
        features = train.prepare_fit_input(np.ones((4, 2), dtype=np.float32), 2, 1, 10, 1e-4)
        self.assertEqual(features.dtype, torch.float64)

        def centers(features, count, rng):
            self.assertEqual(features.dtype, torch.float64)
            return features[:count].clone()

        wrapped = torch_check_adapter(SimpleNamespace(init_centers_kmeanspp=centers))
        self.assertEqual(wrapped.init_centers_kmeanspp(np.ones((4, 2)), 2, None).dtype, np.float64)

    def test_wrong_output_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "float32 tensor"):
            train.validate_tensor(torch.ones((2, 2), dtype=torch.float64), (2, 2), "centers")

    def test_float32_responsibility_roundoff_is_allowed_but_bad_sums_are_not(self):
        X = torch.zeros((2, 1), dtype=torch.float32)
        weights = torch.tensor([.5, .5])
        means, variances = torch.zeros((2, 1)), torch.ones((2, 1))

        def estep(*args):
            return torch.tensor([[.50002, .50002], [.5, .5]]), 1.

        kernels = SimpleNamespace(gmm_estep=estep)
        train.estep_and_validate(X, weights, means, variances, kernels)
        kernels.gmm_estep = lambda *args: (torch.full((2, 2), .51), 1.)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            train.estep_and_validate(X, weights, means, variances, kernels)

    def test_config_defaults_to_float32_and_validates_precision(self):
        import json
        from utils.data import validate_config

        cfg = json.loads((PACKAGE / "config.json").read_text())
        cfg.pop("dtype", None)
        self.assertEqual(validate_config(cfg)["dtype"], "float32")
        self.assertEqual(validate_config(dict(cfg, dtype="float64"))["dtype"], "float64")
        for invalid in ("float16", "bfloat16", 32, None):
            with self.subTest(dtype=invalid), self.assertRaisesRegex(ValueError, "dtype"):
                validate_config(dict(cfg, dtype=invalid))

    def test_default_is_cpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device(), "cpu")

    def test_explicit_device_overrides_environment(self):
        with patch.dict(os.environ, {"LAB2_DEVICE": "cuda"}):
            self.assertEqual(resolve_device("cpu"), "cpu")

    def test_replay_device_is_inherited(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device(inherited="cuda"), "cuda")

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_device("cupy")

    def test_cpu_uses_pytorch(self):
        metadata = validate_device("cpu")
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["torch"], torch.__version__)

    def test_cuda_never_falls_back(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "is_available.*false"):
                validate_device("cuda")

    def test_cuda_metadata(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="GPU"), \
             patch.object(torch.version, "cuda", "12.8"), \
             patch.object(torch, "__version__", "test"):
            self.assertEqual(validate_device("cuda"), dict(device="cuda", torch="test",
                             cuda="12.8", device_name="GPU"))


if __name__ == "__main__":
    unittest.main()
