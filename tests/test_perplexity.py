import unittest
from scripts.evaluate_ppl import paired_retention_interval, wilson_interval, windows


class WindowTests(unittest.TestCase):
    def test_every_target_is_scored_once_including_partial_tail(self):
        for length in (2, 7, 8, 9, 17, 33):
            for stride in (1, 3, 7):
                targets = []
                for start, end, first in windows(length, 8, stride):
                    self.assertLessEqual(end-start, 8)
                    self.assertGreaterEqual(first, start+1)
                    self.assertEqual((end-start-1)-(first-start-1), end-first)
                    targets.extend(range(first, end))
                self.assertEqual(targets, list(range(1, length)))

    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            list(windows(20, 8, 8))

    def test_wilson_interval_contains_observed_accuracy(self):
        low, high = wilson_interval(70, 100)
        self.assertLess(low, .7)
        self.assertGreater(high, .7)

    def test_identical_predictions_retain_all_quality(self):
        values = [True, False, True, True, False] * 20
        low, high = paired_retention_interval(values, values, samples=500)
        self.assertEqual((low, high), (1.0, 1.0))
