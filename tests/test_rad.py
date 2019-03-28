import unittest
import numpy as np
from rad import rad


class TestC(unittest.TestCase):
    """
    Test behavior of the average length of an unsuccessful binary search, c.
    """

    def test_negative_length_throws_exception(self):
        n = np.random.randint(-1000, 0)
        self.assertRaises(ValueError, rad.c, n)

    def test_positive_length_gives_positive_c(self):
        n = np.random.randint(1, 1000)
        self.assertTrue(rad.c(n) >= 1)


class TestS(unittest.TestCase):
    """
    Test behavior of the scoring function, s. We define `x` as the depth of the
    respective node, and `s` as the number of instances (`sample_size`).
    """

    def test_small_x_small_n_is_anomaly(self):

        # limit(x -> 0) and limit(n -> 0) => anomaly; node is near the top
        x = np.random.randint(0, 3)
        n = np.random.randint(5, 10)
        self.assertTrue(rad.s(x=x, n=n) >= .5)

    def test_small_x_large_n_is_anomaly(self):

        # limit(x -> 0) and limit(n -> inf) => anomaly; node if near the top
        x = np.random.randint(0, 3)
        n = np.random.randint(50, 1000)
        self.assertTrue(rad.s(x=x, n=n) >= .5)

    def test_large_x_large_n_is_normal(self):

        # limit(x -> inf) and limit(n -> inf) => normal; node is far down
        x = np.random.randint(10, 1000)
        n = np.random.randint(50, 1000)
        self.assertTrue(rad.s(x=x, n=n) <= .5)

    def test_large_x_small_n_is_normal(self):

        # limit(x -> 0) and limit(n -> inf) => normal; node is far down
        x = np.random.randint(10, 1000)
        n = np.random.randint(50, 100)
        self.assertTrue(rad.s(x=x, n=n) <= .5)

if __name__ == '__main__':
    unittest.main()
