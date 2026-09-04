import math
import random
import unittest
from llm_quantization.reference import quantize_symmetric, pack_int4, unpack_int4


class QuantizationTests(unittest.TestCase):
    def test_known_values(self):
        q = quantize_symmetric([-7, -3, 0, 3, 7], group_size=5)
        self.assertEqual(q.codes, (-7, -3, 0, 3, 7))
        self.assertEqual(q.scales, (1.0,))
        self.assertEqual(q.dequantize(), [-7, -3, 0, 3, 7])

    def test_rounding_error_bound_and_partial_group(self):
        rng = random.Random(42)
        values = [rng.uniform(-5, 5) for _ in range(257)]
        for bits in (4, 8):
            q = quantize_symmetric(values, bits=bits, group_size=32)
            self.assertEqual(len(q.scales), 9)
            for i, (x, restored) in enumerate(zip(values, q.dequantize())):
                self.assertLessEqual(abs(x - restored), q.scales[i // 32] / 2 + 1e-12)

    def test_zero_and_empty(self):
        self.assertEqual(quantize_symmetric([0, 0]).dequantize(), [0, 0])
        self.assertEqual(quantize_symmetric([]).dequantize(), [])

    def test_outlier_localization(self):
        values = [0.1, 0.2, 0.3, 100.0]
        coarse = quantize_symmetric(values, group_size=4).dequantize()
        grouped = quantize_symmetric(values, group_size=2).dequantize()
        self.assertLess(sum((values[i] - grouped[i])**2 for i in (0, 1)),
                        sum((values[i] - coarse[i])**2 for i in (0, 1)))

    def test_invalid_quantization_inputs(self):
        for kwargs in ({'bits': 3}, {'group_size': 0}, {'group_size': 1.5}):
            with self.assertRaises(ValueError):
                quantize_symmetric([1], **kwargs)
        for value in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                quantize_symmetric([value])

    def test_int4_known_bytes(self):
        self.assertEqual(pack_int4([-8, 7, -1]), bytes([0x78, 0x0F]))
        self.assertEqual(unpack_int4(bytes([0x78, 0x0F]), 3), (-8, 7, -1))

    def test_int4_round_trip(self):
        for count in range(34):
            codes = tuple(i % 16 - 8 for i in range(count))
            packed = pack_int4(codes)
            self.assertEqual(len(packed), (count + 1) // 2)
            self.assertEqual(unpack_int4(packed, count), codes)

    def test_invalid_packing(self):
        for codes in ([8], [-9], [1.5]):
            with self.assertRaises(ValueError):
                pack_int4(codes)
        with self.assertRaises(ValueError):
            unpack_int4(b'', 1)


if __name__ == '__main__':
    unittest.main()
