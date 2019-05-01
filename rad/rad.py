"""
Red Hat Anomaly Detection, or RAD, is a python module for performing various
anomaly detection tasks in-support of the AI-Ops effort. RAD leverages the
Isolation Forest (IF) ensemble data-structure; a class that partitions a
data-set and leverages such slicing to gauge magnitude of anomaly. In other
words, the more partitions, the more "normal" the record.
Much of the algorithms in this module are from the works of Liu et al.
(https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
"""

import pickle
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import BytesIO
from scipy.stats import norm
from collections import namedtuple


__version__ = "0.9.7"


# for modeling IsolationForest node instances
Node = namedtuple("Node",
                  ['data', 'size', 'pos', 'value', 'depth', 'left', 'right',
                   'type'])

logging.basicConfig(format="%(asctime)s | "+\
                           "%(funcName)s | "+\
                           "#%(lineno)d | "+\
                           "%(levelname)s | "+\
                           "%(message)s",
                    level=logging.INFO)


def c(n):
    """
    The average length of an unsuccessful binary search query. Note that this
    function is the same as Equation 1 of Isolation Forest manuscript.

    Args:
        n (int): number of records; same as `sample_size` in `IsolationForest`

    Returns:
        average length of an unsuccessful binary search query.
    """
    logging.info("Computing BST length given n={}".format(n))
    if n <= 0:
        raise ValueError("`n` must be positive; length cannot be negative.")
    euler_constant = 0.5772156649
    h = np.log(n) + euler_constant  # Harmonic number
    return 2.*h - (2.*(n-1)/n)


def s(x, n):
    """
    Compute the anomaly score, s. Note that this function is the same as
    Equation 2 of the Isolation Forest manuscript. Such values range between
    0 to 1; those near 0 are deemed "normal", while values near 1 are deemed
    "anomalous". Generally, values > 0.5 are viable anomalous candidates.
    Along these lines, we assert that the smaller `x` is, aka. shorter path,
    the more anomalous. In contrast, the larger `x` is, the longer the path
    since said-record needs more partitions to isolate it on its own.

    Args:
        x (float): scaled depth; equals cumulative depth of data / num_trees
        n (int): number of records; same as `sample_size` in `IsolationForest`

    Returns:
        anomaly score between 0 and 1.
    """
    logging.info("Computing anomaly score given x={}, n={}".format(x, n))
    return 2.0 ** (-x / c(n))


def inventory_data_to_pandas(dic, *args):
    """
    Parse a JSON object, fetched from the Host Inventory Service, and massage
    the data to serve as a pandas DataFrame. We define rows of this DataFrame
    as unique `display_name` instances, and individual column names being an
    individual "system fact" keyword within the `facts` key. Each row-column
    cell is the value for said system-fact.
    To enrich such extraction, you could define `target_features` to explicitly
    define what system facts you wish to extract. If this is not set, this list
    is set to a predefined list of high-level system facts such as CPU count
    and BIOS details.

    Args:
        dic (dict): dictionary from `fetch_fetch_inventory_data(...)`
        args (list): collection of "system_profile" keys wishing to be parsed

    Returns:
        DataFrame: each column is a feature and its cell is its value.
    """

    # do some exception handling to make sure the right data is passed-in
    if not isinstance(dic, dict):
        raise TypeError("`dic` must be of type dict")

    # all system facts branch-off "results"
    if "results" not in dic:
        raise IOError("Ensure `dic` has a `results` key before continuing.")

    # unique set of system facts; for logging purposes only
    unique_keys = set()

    # iterate over the `results` since these contain system facts or profiles
    rows = []
    for i, result in enumerate(dic["results"], start=1):

        # each `result` must have an "id" and "system_profile"
        system_id = result.get("id")
        system_profile = result.get("system_profile")

        # all `result` instances must have an "id" key
        if system_id is None:
            raise IOError("Result number {} lacks an `id` key".format(i))

        # all `result` instances must have a "system_profile" key
        if system_profile is None:
            raise IOError("Result number {} lacks `system_profile`".format(i))

        # all "system_profile" entries must be dictionary data-structures.
        if not isinstance(system_profile, dict):
            raise TypeError("Ensure `system_profile` is a dict.")

        # begin by adding the system identifier
        row = {"id": system_id}

        for key, value in system_profile.items():
            unique_keys.add(key)

            # ensure the system profile is the one desired
            if key not in args and len(args) != 0:
                continue

            # handle system flags which are scalars
            if isinstance(value, (str, bool, float, int)):
                row.update({key: value})

            # handle system flags where all elements are not dict; easy parsing
            elif isinstance(value, list):
                lacks_dict = all(map(lambda x: not isinstance(x, dict), value))
                if lacks_dict:
                    keys = list(map(lambda x: key + "|" + x, value))
                    row.update({}.fromkeys(keys, True))

        # append the individual row
        rows.append(pd.Series(row))

    # concatenate all the individual rows into one singular DataFrame
    logging.info("Facts parsed: {}".format(sorted(unique_keys)))
    frame = pd.concat(rows, axis=1, sort=False).T.set_index("id")
    if frame.size == 0:
        raise ValueError("No data was parsed. Verify data or validity of *args")
    return frame


def preprocess(frame, index=None, drop=None):
    """
    Performs important DataFrame pre-processing so that indices can be set,
    columns can be dropped, or non-numeric columns be encoded as their
    equivalent numeric.

    Args:
        frame (DataFrame): pandas DataFrame.
        index (str or list): columns to serve as DataFrame index.
        drop (str or list): columns to drop from the DataFrame.

    Returns:
         DataFrame and dict: processed DataFrame and encodings of its columns.
    """

    # copy the frame so the original is not overwritten
    logging.info("Data shape: {} x {}".format(*frame.shape))
    df = pd.DataFrame(frame).fillna(0)

    # set the index to be something that identifies each row
    if index is not None:
        df.set_index(index, inplace=True)

    # drop some spurious columns, i.e. `upload_time`
    if drop is not None:
        df.drop(drop, axis=1, inplace=True)

    # encode non-numeric columns as integer; datetimes are not `object`
    mappings = {}
    for column in df.select_dtypes(include=(object, bool)):

        # convert the non-numeric column as categorical and overwrite column
        logging.info("Mapping `{}` to integer".format(column))
        category = df[column].astype("category")
        df[column] = category.cat.codes.astype(float)

        # column categories and add mapping, i.e. "A" => 1, "B" => 2, etc.
        cats = category.cat.categories
        mappings[column] = dict(zip(cats, range(len(cats))))

    # remove all remaining columns, i.e. `datetime`
    df = df.select_dtypes(include=np.number)
    logging.info("# columns encoded as integer: {}".format(len(mappings)))

    # return the DataFrame and categorical mappings
    return df, mappings


class IsolationForest:
    """
    Constructs an ensemble anomaly detection data-structure known as an
    IsolationForest, or IF. How an IF works is by partitioning a user-specified
    data-set into and defining magnitude of anomaly to be inversely proportional
    to the number of slices or partitions. The reasoning here is that an
    anomalous record would require fewer slices to be isolated on its own,
    compared to a "normal" or expected record.
    Much of this logic is inspired by https://github.com/mgckind/iso_forest

    Args:
        array (ndarray): numeric array comprised of N records.
        num_trees (int): number of trees to make in this ensemble.
        sample_size (int): number of records randomly selected per tree.
        limit (int): maximum tree depth.
        seed (int): random number generator seed.
    """
    def __init__(self, array, num_trees=150, sample_size=30, limit=None,
                 seed=None):
        logging.info("Building forest with {} tree(s)".format(num_trees))
        self.num_trees = num_trees
        table, mapping = preprocess(array)
        self.X = table
        self.mapping = mapping
        self.num_records = len(table)
        self.sample_size = sample_size
        self.trees = []
        self.limit = limit
        self.rng = np.random.RandomState(seed)
        self._predictions = None
        logging.info("Sample size and limit: {}".format(sample_size, limit))

        # ensure that the data is truly numeric
        if table.shape != table.select_dtypes(include=np.number).shape:
            raise ValueError("Non-numeric features found. Try `preprocess`.")

        # ensure that sample_size is positive
        if sample_size <= 0:
            raise ValueError("`sample_size` must be positive")

        # set the height limit
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample_size)))
            logging.info("New limit set to {}".format(self.limit))

        # train a tree around a subset of the data, hence ensemble
        for _ in range(self.num_trees):

            # select so-many rows
            logging.info("Sampling {} records".format(self.sample_size))
            ix = self.rng.choice(range(self.num_records), self.sample_size)
            subset = table.values[ix]
            self.trees.append(IsolationTree(subset, 0, self.limit, seed=seed))
            logging.info("{} trees built OK".format(len(self.trees)))

    @staticmethod
    def dump(forest, out_file):
        """
        Persist an IsolationForest instance as either a python pickle object.

        Args:
            forest (IsolationForest): IsolationForest instance.
            out_file (str): output filename.
        """
        if not isinstance(forest, IsolationForest):
            raise ValueError("`forest` must be IsolationForest.")
        with open(out_file, "wb") as handle:
            pickle.dump(forest, handle, protocol=-1)

    @staticmethod
    def dumps(forest):
        """
        Persist an IsolationForest instance as either a Python byte-stream.

        Args:
            forest (IsolationForest): IsolationForest instance.

        Returns:
            byte-stream modeling an IsolationForest instance.
        """
        if not isinstance(forest, IsolationForest):
            raise ValueError("`forest` must be IsolationForest.")
        return pickle.dumps(forest, protocol=-1)

    @staticmethod
    def load(out_file):
        """
        Read-in a persisted IsolationForest instance. Such persistence models
        the object as a Python pickle object.

        Args:
            out_file (pickle): persisted IsolationForest.

        Returns:
            an IsolationForest instance.
        """
        with open(out_file, "rb") as handle:
            forest = pickle.load(handle)
            if not isinstance(forest, IsolationForest):
                raise ValueError("`forest` must be IsolationForest.")
            return forest

    @staticmethod
    def loads(stream):
        """
        Read-in a persisted IsolationForest instance. Such persistence models
        the object as a Python byte-stream.

        Args:
            stream (bytes): byte-stream representation of an IsolationForest.

        Returns:
            an IsolationForest instance.
        """
        forest = pickle.loads(stream)
        if not isinstance(forest, IsolationForest):
            raise ValueError("Argument must model an IsolationForest")
        return forest

    def predict(self, array, min_score=0.5):
        """
        Given a new user-provided array, generate an anomaly score. Such scores
        range from 0 to 1; values near 0 are not anomalous, while values near
        1 would be interesting from an anomaly standpoint.

        Args:
            array (ndarray): numeric array comprised of N records.
            min_score (float): minimum-allowable score to be labeled an anomaly.

        Returns:
            out: array that contains the `id`, `score`, and `depth` per record.
        """
        logging.info("Input query data-set: {} x {}".format(*array.shape))
        data, mapping = preprocess(array)

        # if an ndarray or DataFrame lacking index-name is given, set index name
        if data.index.names == [None]:
            data.index.names = ["id"]

        # for keeping track of anomaly scores
        out = []

        # generate an anomaly score for each row in the dataset, array
        for ix, row in data.iterrows():
            logging.info("Computing score for `{}` across trees".format(ix))

            # for each record, i, find out its depth in each tree, j
            depth = 0
            for j in range(self.num_trees):
                depth += float(TreeScore(row.values, self.trees[j]).path)
            logging.info("Depth: {}".format(depth))

            # scale the depth by the total number of trees
            depth_scaled = depth / self.num_trees
            logging.info("Scaled depth: {}".format(depth_scaled))

            # output a `score` and `depth` per record
            score = s(depth_scaled, self.sample_size)

            # each record (row) has a score and depth
            record = {"score": score,
                      "depth": depth_scaled,
                      "is_anomalous": bool(score > min_score)}
            logging.info("Outcome: {}".format(record))

            # if the index is a MultiIndex each index name index value
            if isinstance(ix, (tuple, list)):
                record.update(zip(data.index.names, ix))

            # otherwise, the index is just an Index, so map this one name
            else:
                record.update({data.index.name: ix})

            # create a centralized data-structure to feed into json.dumps(...)
            out.append(record)
        self._predictions = out
        logging.info("Predictions OK")
        return out

    def predict_and_contrast(self, array, min_score=0.5, alpha=0.05):
        """
        Performs both `predict` and contrasts features which are enriched in
        the anomalous group compared to the "normal" group. Thus, you can
        isolate features that are enriched. The way such contrasting works is
        by deriving the Z-score, (x - u / s), where x is the value of each
        anomalous value, u is the mean of all normal values in said-feature,
        and s is its corresponding standard deviation.

        Args:
            array (ndarray): numeric array comprised of N records.
            min_score (float): minimum-allowable score to be labeled an anomaly.
            alpha (float): p-value cutoff.

        Returns:
            out: array that contains the `id`, `score`, and `depth` per record.
        """

        # generate predictions for this respective data-set
        this_data = pd.DataFrame(array)
        preds = self.predict(this_data, min_score=min_score)

        # join original data with its predictions and group-by anomalous
        logging.info("Joining predictions with features to drive contrasting")
        merged = pd.concat((this_data.reset_index(drop=True),
                            pd.DataFrame(preds)), axis=1)
        logging.info("Merged data-set shape: {:,} x {:,}".format(*merged.shape))

        # group records by whether they are anomalous or not
        agg = merged.groupby("is_anomalous")
        logging.info("Merged group names: {}".format(list(agg.groups.keys())))

        # contrasting requires two groups: those deemed anomalous versus normal
        if len(agg.groups) != 2:
            raise ValueError("Contrast error; add data or increase sample_size")

        # get the "normal" data, i.e. those deemed not anomalous
        normal_subset = agg.get_group(False)
        for i, pred in enumerate(preds):

            # if it is an anomaly, add the features that warrant this label
            if pred["is_anomalous"]:
                logging.info("Contrasting features of anomaly: {}".format(pred))
                its_data = dict(this_data.iloc[i])
                anomalous_features = []

                # per column, derive "normal" point-estimates
                for column, sample_mean in its_data.items():
                    vector = normal_subset[column]
                    pop_mean = vector.mean()
                    pop_std = vector.std()
                    logging.info("Column: {}, x={}".format(column, sample_mean))
                    logging.info("mu={}, sigma={}".format(pop_mean, pop_std))

                    # you cannot divide by zero
                    if pop_std == 0:
                        continue

                    # compute Z score and derive two-tailed p-value
                    z_score = (sample_mean - pop_mean) / pop_std
                    p_value = norm.sf(abs(z_score)) * 2
                    logging.info("p-value: {}".format(p_value))

                    # add significant p-value feature to the prediction
                    if p_value < alpha:
                        a_feature = {"feature": column,
                                     "pvalue": p_value,
                                     "observed_value": sample_mean,
                                     "normal_mean": pop_mean,
                                     "normal_stdev": pop_std
                                     }
                        anomalous_features.append(a_feature)
                        logging.info("Enriched feature: {}".format(a_feature))
                pred["anomalous_features"] = anomalous_features
                pred["num_features"] = len(anomalous_features)
        self._predictions = preds
        logging.info("Predictions and contrast OK")
        return preds

    def to_report(self):

        # read-in predictions as a pandas DataFrame
        preds = self._predictions
        logging.info("Making frame given {} predictions".format(len(preds)))
        frame = pd.DataFrame.from_records(preds)

        # determine those deemed anomalous from normal
        normal = frame[~frame["is_anomalous"]]["score"]
        anomalies = frame[frame["is_anomalous"]]["score"]
        logging.info("# normal predictions: {}".format(len(normal)))
        logging.info("# anomalous predictions: {}".format(len(anomalies)))

        out = {}
        header = "data:image/png;base64,"

        # figure 1 - a boxplot of anomaly score distributions
        logging.info("Creating boxplot given such groups")
        plt.clf()
        plt.boxplot([normal.values, anomalies.values],
                    showfliers=True,
                    showmeans=True)
        plt.xticks([1, 2], ["Normal", "Anomaly"])
        plt.xlabel("Group")
        plt.ylabel("Anomaly Score")

        # model the first plot
        boxplot_io = BytesIO()
        plt.savefig(boxplot_io, format="png")
        boxplot_io.seek(0)
        boxplot_b64 = base64.b64encode(boxplot_io.read())
        out["boxplot"] = header + boxplot_b64.decode("utf-8")

        # figure 2 - a basic histogram of score distributions
        logging.info("Creating histogram given anomaly scores")
        plt.clf()
        histogram_io = BytesIO()
        plt.hist(frame["score"], bins=25)
        plt.ylabel("Count")
        plt.xlabel("Anomaly Score")
        plt.savefig(histogram_io, format="png")
        histogram_io.seek(0)
        histogram_b64 = base64.b64encode(histogram_io.read())
        out["histogram"] = header + histogram_b64.decode("utf-8")

        logging.info("Reporting OK")
        return out


class IsolationTree:
    """
    An individual component of an `IsolationForest`, hence IsolationTree. An
    IsolationForest will have many IsolationTree instances, each modeling a
    subset of the data. Ideally, end-users should not interface with this class
    directly; much of the work is orchestrated through an IsolationForest.

    Args:
        data (ndarray): numeric; X records (X = IsolationForest.sample_size)
        depth (int): the depth of the current object.
        limit (int): maximum limit of the current object.
        seed (int): random number generator seed.
    """
    def __init__(self, data, depth, limit, seed=None):
        self.depth = depth  # depth
        self.data = np.asarray(data)
        self.num_records = len(data)
        logging.info("# records: {:,} and depth: {}".format(len(data), depth))

        # list of N integers where N is the number of features
        self.column_positions = np.arange(self.data.shape[1])
        self.limit = limit  # depth limit
        logging.info("Limit set to: {}".format(self.limit))
        self._value = None

        # a column number or position that is selected; from 0 to N
        self._pos = None
        self.num_external_nodes = 0
        self.num_internal_nodes = 0
        self.rng = np.random.RandomState(seed)
        self.root = self._populate(data, depth, limit)

    def _populate(self, data, depth, l):
        """
        Recursively populate the tree; akin to extension of the tree.
        """

        self.depth = depth
        logging.info("# records to recursively parse: {}".format(len(data)))
        if depth >= l or len(data) <= 1:
            logging.info("Depth reached; at external node")
            left = None
            right = None
            self.num_external_nodes += 1

            # add terminal node (leaf node)
            return Node(data=data,
                        size=len(data),
                        pos=self._pos,
                        value=self._value,
                        depth=depth,
                        left=left,
                        right=right,
                        type='external')
        else:

            # step 1. pick a column number
            self._pos = self.rng.choice(self.column_positions)  # pick a column
            logging.info("Column number selected: {:,}".format(self._pos))

            # step 2. select the minimum and maximum values in said-column
            min_ = data[:, self._pos].min()  # get min value from the column
            max_ = data[:, self._pos].max()  # get max value from the column
            logging.info("Column min and max: {:,}...{:,}".format(min_, max_))
            if min_ == max_:
                logging.info("Min and max are equal; at external node")

                # if extrema are equal, such nodes lack descendants
                left = None
                right = None
                self.num_external_nodes += 1
                return Node(data=data,
                            size=len(data),
                            pos=self._pos,
                            value=self._value,
                            depth=depth,
                            left=left,
                            right=right,
                            type='external')

            # step 3. generate a random number between the min and max range
            self._value = self.rng.uniform(min_, max_)
            logging.info("Real value between extrema: {}".format(self._value))

            # step 4. determine if values in said-column are less than the value
            truth = np.where(data[:, self._pos] < self._value, True, False)

            # `left` are where values are less than value, `right` otherwise
            left = data[truth]
            right = data[~truth]
            logging.info("# records as left node: {}".format(len(left)))
            logging.info("# records as right node: {}".format(len(right)))

            # recursively repeat by propogating the left and right branches
            self.num_internal_nodes += 1
            return Node(data=data,
                        size=len(data),
                        pos=self._pos,
                        value=self._value,
                        depth=depth,
                        left=self._populate(left, depth + 1, l),
                        right=self._populate(right, depth + 1, l),
                        type='internal')


class TreeScore(object):
    """
    A helper-class that is leveraged when executing IsolationForest.predict().
    This class works by taking a user-provided record, `vector`, and measuring
    its depth with respect to one of the IsolationTree, `tree`, instances in
    the forest.

    Args:
        vector (list): a D-dimensional array.
        tree (IsolationTree): instance of type IsolationTree.
    """
    def __init__(self, vector, tree):
        self.vector = vector
        self.depth = 0
        self.path = self._traverse(tree.root)

    def _traverse(self, node):
        """
        Recursively move down the current tree and enumerate its level or depth.
        When traversal is complete, the depth is quantified given c(...), and
        will be aggregated with results from other IsolationTree instances.

        Args:
            node (Node): the current tree node; see `Node` for details.
        """

        # at the end of recursion, return the number of levels traversed
        if node.type == 'external':
            if node.size == 1:
                return self.depth
            else:
                self.depth = self.depth + c(node.size)
                return self.depth
        else:
            attrib = node.pos  # the attribute
            self.depth += 1  # increment the level

            # if value is less than the nodes value, go left
            if self.vector[attrib] < node.value:
                return self._traverse(node.left)

            # if value is greater than the nodes value, go right
            else:
                return self._traverse(node.right)
