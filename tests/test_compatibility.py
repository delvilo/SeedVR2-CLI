"""Unit tests for src/optimization/compatibility.py GGUF availability validation."""

import unittest
from unittest.mock import MagicMock, patch

from src.optimization.compatibility import validate_gguf_availability


class TestGGUFAvailabilityValidation(unittest.TestCase):
    def test_gguf_available_does_not_raise(self):
        with patch("src.optimization.compatibility.GGUF_AVAILABLE", True):
            # Test default operation
            validate_gguf_availability()
            # Test custom operation
            validate_gguf_availability(operation="dequantize GGUF model")

    def test_gguf_not_available_raises_runtime_error(self):
        with patch("src.optimization.compatibility.GGUF_AVAILABLE", False):
            # Test default operation
            with self.assertRaises(RuntimeError) as ctx:
                validate_gguf_availability()
            self.assertIn("GGUF library required to load GGUF model", str(ctx.exception))

            # Test custom operation
            with self.assertRaises(RuntimeError) as ctx:
                validate_gguf_availability(operation="load custom GGUF model")
            self.assertIn("GGUF library required to load custom GGUF model", str(ctx.exception))

    def test_gguf_not_available_logs_error_when_debug_provided(self):
        with patch("src.optimization.compatibility.GGUF_AVAILABLE", False):
            mock_debug = MagicMock()
            with self.assertRaises(RuntimeError):
                validate_gguf_availability(operation="load GGUF model", debug=mock_debug)

            mock_debug.log.assert_called_once()
            args, kwargs = mock_debug.log.call_args
            self.assertIn("Cannot load GGUF model: GGUF library is not installed.", args[0])
            self.assertEqual(kwargs.get("level"), "ERROR")
            self.assertEqual(kwargs.get("category"), "setup")
            self.assertEqual(kwargs.get("force"), True)


if __name__ == "__main__":
    unittest.main()
