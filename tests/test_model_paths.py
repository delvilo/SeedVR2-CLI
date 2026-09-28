"""Model location behavior needed for persistent Colab Drive storage."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.constants import find_model_file, get_base_cache_dir


class ModelPathTests(unittest.TestCase):
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
