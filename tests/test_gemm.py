import unittest

import numpy as np
import micro_llm_engine as mle


class TestFp32Gemm(unittest.TestCase):
    def test_matches_numpy_for_various_matrix_shapes(self):
        rng = np.random.default_rng(seed=0)

        for m, k, n in (
            (1, 1, 1),
            (2, 3, 4),
            (7, 5, 2),
            (16, 32, 8),
            (128, 256, 192),
            (384, 256, 320),
        ):
            with self.subTest(shape_a=(m, k), shape_b=(k, n)):
                a = rng.standard_normal((m, k)).astype(np.float32)
                b = rng.standard_normal((k, n)).astype(np.float32)

                actual = mle.fp32_gemm(a, b)
                expected = a @ b

                self.assertEqual(actual.shape, (m, n))
                self.assertEqual(actual.dtype, np.float32)
                atol = max(1e-6, k * np.finfo(np.float32).eps)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=atol)

    def test_accepts_non_contiguous_inputs(self):
        rng = np.random.default_rng(seed=1)
        a = rng.standard_normal((4, 6)).astype(np.float32)[:, ::2]
        b = rng.standard_normal((6, 8)).astype(np.float32)[::2, :]

        actual = mle.fp32_gemm(a, b)
        expected = a @ b

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rejects_incompatible_dimensions(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((4, 2), dtype=np.float32)

        with self.assertRaises(RuntimeError):
            mle.fp32_gemm(a, b)


if __name__ == "__main__":
    unittest.main()