import unittest
import numpy as np


class TestUtilityInequalities(unittest.TestCase):
    def test_upper_bound_on_quad(self):
        """
        Tests x / (c + a - b) <= (c + b) / (c + a) * x / c for c>0, a=>b=>0, x=>0.
        """
        for _ in range(100):
            b = np.square(np.random.randn(1))
            a = b + np.square(np.random.randn(1))
            c = np.square(np.random.randn(1))
            x = np.square(np.random.randn(1))

            gt = x / (c + a - b)
            ov = x * (c + b) / (c + a) / c
            if ov - gt < 0:
                print((a, b, c, x))
            self.assertGreaterEqual(ov, gt)


if __name__ == '__main__':
    unittest.main()
