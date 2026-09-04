import unittest
from scripts.benchmark_mlx import summarize


class SummaryTests(unittest.TestCase):
    def test_median_and_inclusive_iqr_resist_one_outlier(self):
        keys = ('ttft_seconds', 'decode_tokens_per_second', 'total_seconds', 'peak_mlx_bytes')
        rows = [{k: v for k in keys} for v in [1, 2, 3, 4, 100]]
        for stats in summarize(rows).values():
            self.assertEqual(stats, {'median': 3, 'min': 1, 'max': 100, 'q1': 2, 'q3': 4})

    def test_empty_measurements_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])
