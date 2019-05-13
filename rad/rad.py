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
from sklearn.ensemble import IsolationForest


__version__ = "0.9.9"


logging.basicConfig(format="%(asctime)s | "+\
                           "%(funcName)s | "+\
                           "#%(lineno)d | "+\
                           "%(levelname)s | "+\
                           "%(message)s",
                    level=logging.WARNING)


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
        logging.debug("{} w/{} facts(s)".format(system_id, len(system_profile)))

        for key, value in system_profile.items():
            unique_keys.add(key)

            # ensure the system profile is the one desired
            if args and key not in args:
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


class RADIsolationForest(IsolationForest):

    def __init__(self, **kwargs):
        super(RADIsolationForest, self).__init__(**kwargs)

    def _set_oob_score(self, X, y):
        super(RADIsolationForest, self)._set_oob_score(X, y)

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

    def fit_predict(self, X):
        data = pd.DataFrame(X)
        self.fit(data)
        labels = super(RADIsolationForest, self).predict(data)
        labels = np.where(labels == -1, True, False)
        scores = self.score_samples(data)
        out = []
        if data.index.name is None:
            data.index.name = "id"

        for i in range(len(X)):
            record = {"score": float(scores[i]),
                      "is_anomalous": bool(labels[i])}
            ix = data.index[i]

            # if the index is a MultiIndex each index name index value
            if isinstance(ix, (tuple, list)):
                record.update(zip(data.index.names, ix))
            else:
                record.update({data.index.name: ix})
            out.append(record)
        return out

    def fit_predict_contrast(self, X, training_frame, alpha=0.05):

        # generate predictions for this respective data-set
        this_data = pd.DataFrame(training_frame)
        preds = self.fit_predict(X)

        # join original data with its predictions and group-by anomalous
        merged = pd.concat((this_data.reset_index(drop=True),
                            pd.DataFrame(preds)), axis=1)

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
                    # if pop_std == 0:
                    #     continue

                    # compute Z score and derive two-tailed p-value
                    z_score = (sample_mean - pop_mean) / max((pop_std, 1))
                    p_value = norm.sf(abs(z_score)) * 2
                    logging.info("p-value: {}".format(p_value))

                    # add significant p-value feature to the prediction
                    if p_value < alpha:
                        a_feature = {"feature": column,
                                     "pvalue": float(p_value),
                                     "observed_value": float(sample_mean),
                                     "normal_mean": float(pop_mean),
                                     "normal_stdev": float(pop_std)
                                     }
                        anomalous_features.append(a_feature)
                        logging.info("Enriched feature: {}".format(a_feature))
                pred["anomalous_features"] = anomalous_features
                pred["num_features"] = len(anomalous_features)
        logging.info("Predictions and contrast OK")
        return preds
