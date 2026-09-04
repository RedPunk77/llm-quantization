import unittest
from scripts.evaluate_ppl import windows


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
