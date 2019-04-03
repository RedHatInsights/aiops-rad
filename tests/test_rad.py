import unittest
import numpy as np
import pandas as pd
from rad.rad import IsolationForest, IsolationTree

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


class TestPreprocessOn(unittest.TestCase):

    def setUp(self):
        size = np.random.randint(1, 100, size=2)
        data = np.random.randint(-100000, 100000, size)
        self.frame = pd.DataFrame(data)
        self.frame["groups"] = np.random.choice(["a", "b"], len(self.frame))

    def test_preprocess_on_output_equals_num_groups(self):
        """
        Test that the number of chunks from `preprocess_on` equals the number
        of groups, i.e. ["A", "B", "C", ...]
        """
        uniq_groups = np.unique(self.frame["groups"])
        chunks = rad.preprocess_on(self.frame, on="groups", min_records=0)
        self.assertEqual(len(chunks), len(uniq_groups))

    def test_preprocess_on_group_must_exist(self):
        """
        Test that an invalid column set for `on` raises KeyError exception
        """
        self.assertRaises(KeyError, rad.preprocess_on, self.frame, "bad column")


class TestIsolationForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Building an IsolationForest *can* be expensive, so do it only once
        """
        size = np.random.randint(10, 100, size=2)
        data = np.random.randint(0, 10000, size=size)
        cls.forest = IsolationForest(data)

    def test_correct_number_of_trees_made(self):
        """
        Test that if N trees are desired, N trees shall be made
        """
        self.assertEqual(self.forest.num_trees, len(self.forest.trees))

    def test_predict_length_equals_input_length(self):
        """
        Test that each input record has a corresponding prediction
        """
        num_rows = np.random.randint(1, 100)
        data = np.random.randint(0, 10000, (num_rows, self.forest.X.shape[1]))
        out = self.forest.predict(data)
        self.assertEqual(len(out), len(data))

    def test_columns_are_in_contrast(self):
        """
        Test that `contrast` requires same number of columns as for training
        """
        columns = self.forest.X.columns
        new_data = np.random.randint(0, 10000, (100, self.forest.X.shape[1]))
        contrast = self.forest.contrast(new_data)
        self.assertEqual(len(contrast.loc[columns]), self.forest.X.shape[1])


class TestIsolationTree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Building an IsolationTree is less expensive than forest, but do it once
        """
        size = np.random.randint(10, 50, size=2)
        data = np.random.randint(0, 10000, size=size)
        cls.tree = IsolationTree(data, depth=1, limit=10)

    def test_tree_built_some_nodes(self):
        """
        Test that an IsolationTree has a positive number of nodes
        """
        num_nodes = self.tree.num_internal_nodes + self.tree.num_external_nodes
        self.assertTrue(num_nodes > 0)

    def test_random_value_in_column(self):
        """
        Test that given a column, q, a random number, p, falls within its range.
        """
        column = self.tree.data[:, self.tree._pos]
        self.assertTrue(min(column) <= self.tree._value <= max(column))


if __name__ == '__main__':
    unittest.main()
