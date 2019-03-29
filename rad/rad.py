"""
Red Hat Anomaly Detection, or RAD, is a python module for performing various
anomaly detection tasks in-support of the AI-Ops effort. RAD leverages the
Isolation Forest (IF) ensemble data-structure; a class that partitions a
data-set and leverages such slicing to gauge magnitude of anomaly. In other
words, the more partitions, the more "normal" the record.
Much of the algorithms in this module are from the works of Liu et al.
(https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
"""


import os
import s3fs
import pickle
import urllib3
import logging
import requests
import numpy as np
import pandas as pd

from pyarrow import parquet
from requests.auth import HTTPBasicAuth
from collections import namedtuple


__version__ = "0.8.2"


# for modeling IsolationForest node instances
Node = namedtuple("Node",
                  ['data', 'size', 'pos', 'value', 'depth', 'left', 'right',
                   'type'])


def c(n):
    """
    The average length of an unsuccessful binary search query. Note that this
    function is the same as Equation 1 of Isolation Forest manuscript.

    Args:
        n (int): number of records; same as `sample_size` in `IsolationForest`

    Returns:
        average length of an unsuccessful binary search query.
    """
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
    return 2.0 ** (-x / c(n))


def fetch_s3(bucket, profile_name=None, folder=None, date=None,
             endpoint=None, workers=None):
    """
    Queries data collected from Insights that is saved in S3. It is presumed
    `profile_name` (your ~/.aws/credentials name) exhibits credentials to
    facilitate such an access.

    Args:
         endpoint (str): S3 endpoint.
         profile_name (str): AWS credentials; found in ~/.aws/credentials
         bucket (str): S3 bucket name.
         folder (str): folder name; contains many parquet files
         date (str): S3 prefix; is that which is prepended to `bucket`.
         workers (int): maximum number of worker threads.
    """
    if not profile_name:
        profile_name = "default"

    if not endpoint:
        endpoint = "https://s3.upshift.redhat.com"

    if not date:
        date = ""

    if not folder:
        folder = ""

    fs = s3fs.S3FileSystem(profile_name=profile_name,
                           client_kwargs={"endpoint_url": endpoint})

    # concatenate the bucket and all subsequent variables to give a full path
    path = os.path.join(bucket, date, folder)
    obj = parquet.ParquetDataset(path, fs, metadata_nthreads=workers)
    frame = obj.read_pandas().to_pandas()
    return frame


def fetch_inventory_data(email, password, url=None):
    """
    Trivial function to fetch some Host Inventory Data.

    Args:
        email (str): user authentication key.
        password (str): password; defaults to `redhat`
        url (str): endpoint for the host inventory data.

    Returns:
        dict following retrieval from the Host Inventory API.

    Examples:
        >>> dic = fetch_inventory_data()
    """
    if url is None:
        url = "https://ci.cloud.paas.upshift.redhat.com/api/inventory/v1/hosts"

    # it is presumed certificates are not needed to access the URL
    urllib3.disable_warnings()
    resp = requests.get(url, auth=HTTPBasicAuth(email, password), verify=False)
    return resp.json()


def inventory_data_to_pandas(dic):
    """
    Parse a JSON object, fetched from the Host Inventory Service, and massage
    the data to serve as a pandas DataFrame. We define rows of this DataFrame
    as unique `display_name` instances, and individual column names being an
    individual "system fact" keyword within the `facts` key. Each row-column
    cell is the value for said system-fact.

    Args:
        dic (dict): dictionary from `fetch_fetch_inventory_data(...)`

    Returns:
        DataFrame: each column is a feature and its cell is its value.

    Examples:
        >>> dic = fetch_inventory_data()  # provide your specific credentials
        >>> frame = inventory_data_to_pandas(dic)
    """

    # keep track of systems lacking data; useful for finding anomalous signals
    lacks_data = []

    # list of dictionary items for each and every row
    rows = []

    # iterate over all records; all data resides under the `results` key
    for result in dic["results"]:

        # assert that `facts` and `account` are keys, otherwise throw error
        if "facts" not in result:
            raise IOError("JSON must contain `facts` key under `results`")
        if "account" not in result:
            raise IOError("JSON must contain `account` key under `results`")

        # get some preliminary data
        data = result["facts"]
        name = result["display_name"]

        # identify systems which lack data
        if len(data) == 0:
            lacks_data.append(name)
            continue

        # data looks like this:
        # [{'facts': {'fqdn': 'eeeg.lobatolan.home'}, 'namespace': 'inventory'}]

        # iterate over all the elements in the list; usually gets one element
        for dic in data:
            if not isinstance(dic, dict):
                raise IOError("Data elements must be dict")

            if "facts" not in dic:
                raise KeyError("`facts` key must reside in the dictionary")

            # iterate over all the key-value pairs
            for k, v in dic["facts"].items():

                # handling numeric values
                if isinstance(v, (int, bool)):
                    v = float(v)
                    rows.append({"ix": name, "value": v, "col": k})

                # if a collection, each collection item is its own feature
                elif isinstance(v, (list, tuple)):
                    for v_ in v:
                        rows.append({"ix": name,
                                     "value": True,
                                     "col": "{}|{}".format(k, v_)})

                # handling strings is trivial
                elif isinstance(v, str):
                    rows.append({"ix": name,
                                 "value": v,
                                 "col": k})

                # sometimes, values are `dict`, so handle accordingly
                elif isinstance(v, dict):
                    for k_, v_ in v.items():
                        rows.append({"ix": name,
                                     "value": v_,
                                     "col": "{}".format(k_)})

                # end-case; useful if value is None or NaN
                else:
                    rows.append({"ix": name, "value": -1, "col": k})

    # take all the newly-added data and make it into a DataFrame
    frame = pd.DataFrame(rows).drop_duplicates()

    # add all the data that lack values
    for id_ in lacks_data:
        frame = frame.append(pd.Series({"ix": id_}), ignore_index=True)

    frame = frame.pivot(index="ix", columns="col", values="value")
    return frame.drop([np.nan], axis=1)


def preprocess(frame, index=None, drop=None, fill_value=-1):
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
    df = pd.DataFrame(frame).fillna(fill_value)

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
        category = df[column].astype("category")
        df[column] = category.cat.codes.astype(float)

        # column categories and add mapping, i.e. "A" => 1, "B" => 2, etc.
        cats = category.cat.categories
        mappings[column] = dict(zip(cats, range(len(cats))))

    # remove all remaining columns, i.e. `datetime`
    df = df.select_dtypes(include=np.number)

    # return the DataFrame and categorical mappings
    return df, mappings


def preprocess_on(frame, on, min_records=50, index=None, drop=None,
                  fill_value=-1):
    """
    Similar to `preprocess` but groups records in the DataFrame on a group pf
    features. Each respective chunk or block is then added to a list; analogous
    to running `preprocess` on a desired subset of a DataFrame.

    Args:
        frame (DataFrame): pandas DataFrame
        on (str or list): features in `frame` you wish to group around.
        min_records (int): minimum number of rows each grouped chunk must have.
        index (str or list): columns to serve as DataFrame index.
        drop (str or list): columns to drop from the DataFrame.

    Returns:
         DataFrame and dict: processed DataFrame and encodings of its columns.
    """

    data = pd.DataFrame(frame)
    out = []

    # group-by `on` and return the chunks which satisfy minimum length
    for _, chunk in data.groupby(on):
        if len(chunk) > min_records:

            # if only `on` is provided, set this as the index
            if index is None and on is not None:
                index = on

            # run `preprocess` on each chunk
            chunk, mapping = preprocess(chunk, index=index, drop=drop,
                                        fill_value=fill_value)
            out.append((chunk, mapping))
    return out


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
        self.num_trees = num_trees
        self.X = pd.DataFrame(array)
        self.num_records = len(array)
        self.sample_size = sample_size
        self.trees = []
        self.limit = limit
        self.rng = np.random.RandomState(seed)

        # ensure that the data is truly numeric
        if self.X.shape != self.X.select_dtypes(include=np.number).shape:
            raise ValueError("Non-numeric features found. Try `preprocess`.")

        # set the height limit
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample_size)))

        # train a tree around a subset of the data, hence ensemble
        for _ in range(self.num_trees):

            # select so-many rows
            ix = self.rng.choice(range(self.num_records), self.sample_size)
            subset = self.X.values[ix]
            self.trees.append(IsolationTree(subset, 0, self.limit, seed=seed))

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

    def predict(self, array):
        """
        Given a new user-provided array, generate an anomaly score. Such scores
        range from 0 to 1; values near 0 are not anomalous, while values near
        1 would be interesting from an anomaly standpoint.

        Args:
            array (ndarray): numeric array comprised of N records.

        Returns:
            DataFrame: Nx2 array where column 1 and 2 is depth and score.
        """
        data = pd.DataFrame(array)

        # for keeping track of anomaly scores
        scores = []

        # generate an anomaly score for each row in the dataset, array
        for i in range(len(data)):

            # for each record, i, find out its depth in each tree, j
            depth = 0
            for j in range(self.num_trees):
                depth += float(TreeScore(data.values[i], self.trees[j]).path)

            # scale the depth by the total number of trees
            depth_scaled = depth / self.num_trees

            # output a `score` and `depth` per record
            score = s(depth_scaled, self.sample_size)
            scores.append([score, depth_scaled])

        # return anomaly score and depth for each record
        out = pd.DataFrame(scores,
                           index=data.index,
                           columns=["score", "depth"])

        # sort by score so that dissemination can be easier
        return out.sort_values("score", ascending=False)

    def contrast(self, array, min_score=0.55):
        """
        Contrasts features given anomalous and non-anomalous records. This
        would be helpful when diving into "how come" a record was deemed
        anomalous.

        Args:
            array (ndarray): numeric array comprised of N records.
            min_score (float): score cutoff to classify record as anomalous

        Returns:
            DataFrame: table showing various values for anomalous records.
        """

        report = []
        for frame, name in [(self.X, "History"), (array, "Query")]:
            preds = self.predict(frame)
            preds["is anomalous"] = preds["score"] > min_score
            preds["data format"] = name
            combined = pd.concat((pd.DataFrame(frame), preds), axis=1)
            report.append(combined)

        return pd.concat(report).\
            groupby(["is anomalous", "data format"]).\
            mean().T


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

        # list of N integers where N is the number of features
        self.column_positions = np.arange(self.data.shape[1])
        self.limit = limit  # depth limit
        self._value = None

        # a column number or position that is selected; from 0 to N
        self._pos = None
        self.num_external_nodes = 0
        self.rng = np.random.RandomState(seed)
        self.root = self._populate(data, depth, limit)

    def _populate(self, data, depth, l):
        """
        Recursively populate the tree; akin to extension of the tree.
        """

        self.depth = depth
        if depth >= l or len(data) <= 1:
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

            # step 2. select the minimum and maximum values in said-column
            min_ = data[:, self._pos].min()  # get min value from the column
            max_ = data[:, self._pos].max()  # get max value from the column
            if min_ == max_:

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

            # step 4. determine if values in said-column are less than the value
            truth = np.where(data[:, self._pos] < self._value, True, False)

            # `left` are where values are less than value, `right` otherwise
            left = data[truth]
            right = data[~truth]

            # recursively repeat by propogating the left and right branches
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
