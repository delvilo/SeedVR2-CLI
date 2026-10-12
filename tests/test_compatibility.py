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


class TestSparseSageAttentionCompatibility(unittest.TestCase):
    def test_sparse_sage_attention_validation(self):
        from src.optimization.compatibility import validate_attention_mode, SPARSE_SAGE_AVAILABLE
        mode = validate_attention_mode("sparse_sage_attention")
        self.assertIn(mode, ("sparse_sage_attention", "sdpa"))

    def test_call_sparse_sage_attn_varlen_fallback(self):
        import torch
        from src.optimization.compatibility import call_sparse_sage_attn_varlen
        q = torch.randn(8, 2, 16)
        k = torch.randn(8, 2, 16)
        v = torch.randn(8, 2, 16)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
        out = call_sparse_sage_attn_varlen(q, k, v, cu_seqlens, cu_seqlens, 4, 4)
        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.isnan(out).any().item())

    def test_requirements_no_sageattention(self):
        import os
        req_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
        with open(req_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        self.assertNotIn("sageattention", content)
        self.assertIn("triton", content)

if __name__ == "__main__":
    unittest.main()
