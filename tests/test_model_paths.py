"""Model location behavior needed for persistent Colab Drive storage."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.constants import find_model_file, get_base_cache_dir
from src.utils.model_registry import MODEL_REGISTRY


class ModelPathTests(unittest.TestCase):
    def test_registry_uses_canonical_filenames_and_distinct_hashes(self):
        seen = set()
        for name, info in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                self.assertEqual(name, name.strip())
                self.assertEqual(name, Path(name).name)
                self.assertRegex(info.sha256, r"^[0-9a-f]{64}$")
                self.assertNotIn(info.sha256, seen)
                seen.add(info.sha256)

    def test_q8_download_identities_match_publisher_metadata(self):
        # SHA-256 values from the linked Hugging Face file pages in docs/memory.md.
        # Guards the 3B/7B and Q4/Q8 hash mixups introduced by 23d3845.
        expected = {
            "seedvr2_ema_3b-Q8_0.gguf": (
                "AInVFX/SeedVR2_comfyUI", "3B",
                "be0d60083a2051a265eb4b77f28edf494e6db67ffc250216f32b72292e5cbd96",
            ),
            "seedvr2_ema_7b-Q8_0.gguf": (
                "cmeka/SeedVR2-GGUF", "7B",
                "669788655e8f15f306284f267a444e9766c8a421869577b16a961e43029c737b",
            ),
            "seedvr2_ema_7b_sharp-Q8_0.gguf": (
                "cmeka/SeedVR2-GGUF", "7B",
                "b1f81cb5700b0b1f432f2c785528356c952c41c74d03d205c6f14b0bd6da303d",
            ),
        }
        for name, (repo, size, digest) in expected.items():
            with self.subTest(model=name):
                info = MODEL_REGISTRY[name]
                self.assertEqual((info.repo, info.size, info.precision, info.sha256),
                                 (repo, size, "Q8_0", digest))

    def test_explicit_model_directory_takes_precedence(self):
        with tempfile.TemporaryDirectory() as root:
            default = Path(root, "default")
            selected = Path(root, "drive", "models")
            default.mkdir()
            selected.mkdir(parents=True)
            (default / "weights.safetensors").touch()
            (selected / "weights.safetensors").touch()

            with patch.dict(os.environ, {"SEEDVR2_MODEL_DIR": str(default)}):
                self.assertEqual(get_base_cache_dir(), str(default))
                self.assertEqual(
                    find_model_file("weights.safetensors", str(selected)),
                    str(selected / "weights.safetensors"),
                )


if __name__ == "__main__":
    unittest.main()
