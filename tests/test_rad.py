import logging
import unittest
import numpy as np
import pandas as pd

from rad import rad


# disable logging unless error is significant
logging.disable(logging.ERROR)


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        size = np.random.randint(1, 100, size=2)
        data = np.random.randint(-100000, 100000, size)
        self.frame = pd.DataFrame(data)

    def test_preprocess_doesnt_change_numeric_array(self):
        frame, _ = rad.preprocess(self.frame)
        # numeric arrays, after `preprocess`, are exactly the same as before.
        self.assertEqual(frame.values.all(), self.frame.values.all())

    def test_empty_mapping_given_numeric_array(self):
        """
        Test that if a numeric array is given, no mappings are returned.
        """
        _, mapping = rad.preprocess(self.frame)
        self.assertEqual(len(mapping), 0)

    def test_populated_mapping_given_nonnumeric_array(self):
        """
        Test that all string columns get mapped to its respective integer.
        """
        shape = (len(self.frame), np.random.randint(1, 5))
        nd_array = np.random.choice(["a", "b"], shape)
        frame, mapping = rad.preprocess(nd_array)
        self.assertEqual(len(mapping), nd_array.shape[1])

    def test_invalid_column_as_index_raises_keyerror(self):
        """
        Test that an invalid column cannot be set as the index
        """
        self.assertRaises(KeyError, rad.preprocess, self.frame, ["invalid"])

    def test_valid_column_set_as_index(self):
        """
        Test that the index is a valid column name
        """
        column = np.random.choice(self.frame.columns.values)
        frame, mapping = rad.preprocess(self.frame, index=column)
        self.assertEqual(frame.index.name, column)


if __name__ == '__main__':
    unittest.main()
