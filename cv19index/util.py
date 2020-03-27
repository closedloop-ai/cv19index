import math

import numpy as np
import pandas as pd

_defaultDtypeDict = {
    "string": "object",
    "struct": "object",
    "boolean": "bool",
    "double": "float64",
    "integer": "int32",
    "long": "int64",
    "date": "object",
    "timestamp": "object",
    "datetime": "object",
    "array": "object",
}


class UserException(Exception):
    """Raise for exceptions that indicate bad input """


def schema_dtypes(schema_json: list, dtype_dict=None):
    if dtype_dict is None:
        dtype_dict = _defaultDtypeDict

    def lookup(x):
        t = dtype_dict.get(x["dataType"]["dataType"])
        assert len(t) > 0
        return t

    return {x["name"]: lookup(x) for x in schema_json}


def nonnull_column(s: pd.Series):
    """ Takes a pandas Series and return sa vector of all entries that have 
    values (are not None or float.nan).
    """
    if s.dtype == float:
        return np.logical_not(np.isnan(s))
    else:
        return np.logical_not(pd.isnull(s))


def none_or_nan(v):
    """Returns True if the value is None or float('nan')"""
    return v is None or (isinstance(v, float) and math.isnan(v))
