# IO functions to read and write model and prediction files in an appropriate format.

import json
import math
import pickle
from typing import Dict
import pandas as pd
from pandas.io.common import _NA_VALUES
from pkg_resources import resource_filename

from .util import schema_dtypes


INDEX = 'personId'


def read_model(fpath):
    with open(fpath, "rb") as fobj:
        return pickle.load(fobj)


def validate_df(df, dtype) -> pd.DataFrame:
    """ Now verify that everything is exactly as it should be. """
    print(df.columns, dtype)
    incorrect_types = [
        f"{n} expected {t} but was {df[n].dtype}"
        for n, t in dtype.items()
        if n != INDEX and df[n].dtype != t
    ]
    assert len(incorrect_types) == 0, f"Incorrect types:\n" + "\n".join(incorrect_types)

    return df


def read_claim(fpath: str) -> pd.DataFrame:
    schame_fpath = resource_filename("cv19index","resources/xgboost/claims.schema.json")
    return read_frame(fpath, schame_fpath)


def read_demo(fpath: str) -> pd.DataFrame:
    schema_fpath = resource_filename("cv19index", "resources/xgboost/demo.schema.json")
    return read_frame(fpath, schema_fpath)


def read_frame(fpath, schema_fpath) -> pd.DataFrame:
    XLS = ('xls', 'xlsx', 'xlsm', 'xlsb', 'odf')

    def is_excel(fpath: str) -> bool:
        return fpath.endswith(XLS)

    with open(schema_fpath) as f:
        schema = json.load(f)
    dtype = schema_dtypes(schema["schema"])

    if is_excel(fpath):
        df = read_excel(fpath, dtype)
    elif fpath.endswith(".parquet"):
        df = read_parquet(fpath, dtype)
    elif fpath.endswith(".csv"):
        df = read_csv(fpath, dtype)
    else:
        raise TypeError(f'This script reads files based the extension.\n'
                        f'The known extensions are {", ".join(XLS)}, .parquet, .csv'
                        f'Please ensure your file is one of those file types with correct file extension.')

    return df


def read_excel(fpath, dtype) -> pd.DataFrame:
    print(fpath, INDEX, dtype)
    df = pd.read_excel(fpath, dtype=dtype, index_col=INDEX)

    print(df.head(10))

    return validate_df(df, dtype)


def read_parquet(fpath, dtype) -> pd.DataFrame:
    df = pd.read_parquet(fpath)
    # Now set the index.
    df = df.set_index(INDEX)
    return validate_df(df, dtype)


def read_csv(fpath: str, dtype: Dict) -> pd.DataFrame:

    date_cols = [k for k, v in dtype.items() if v == "datetime"]

    # Pandas converts a string with "NA" as a real NaN/Null. We don't want this
    # for real string columns. NA can show up as a real flag in customer data and
    # it doesn't mean it should be treated as NaN/Null.
    def na_vals(x):
        # For now, only ignore NA conversion for strings. Structs/etc can still use it.
        if x["dataType"]["dataType"] in ("string"):
            return []
        else:
            return _NA_VALUES

    na_values = {k: na_vals(v) for k, v in dtype.items()}

    # Pandas read_csv ignores the dtype for the index column,
    # so we don't read in with an index column.  Set it later.
    df = pd.read_csv(
        fpath,
        header=0,
        names=list(dtype.keys()),
        dtype=dtype,
        parse_dates=date_cols,
        na_values=na_values
    )

    # Now go back and fix all the ints (which we read in as floats to handle NAs)
    df = df.fillna(0)
    for k, x in dtype.items():
        if x == "float64":
            df[k] = df[k].astype('int64')

    # Sometimes pandas doesn't pay attention to the type we give it.
    # Go back and ask it to really make every column be what we asked.
    for n, t in dtype.items():
        df[n] = df[n].astype(t)

    # Oh, and if there are date/datetime types that are empty, pandas "helps us out"
    # by converting those to an int(0) instead of None.
    # Thanks pandas, but I'd prefer None here.
    for c in date_cols:
        df.loc[df[c] == 0, [c]] = None

    # Now set the index.
    df = df.set_index(INDEX, drop=True)
    return validate_df(df, dtype)


def write_predictions(predictions, fpath):
    if fpath.endswith(".csv"):
        output = predictions.to_csv(index=False, float_format="%f")
    elif fpath.endswith(".json"):
        if (
            predictions.index.dtype == "object"
            and predictions.index.size > 0
            and type(predictions.index[0]) == list
        ):
            predictions.index = [tuple(l) for l in predictions.index]
        js_rec = predictions.to_json(orient="records", double_precision=3)
        output = f'{{"records":{js_rec}}}'
    elif fpath.endswith(".jsonl"):
        # These next two lines are because there was a bug where to_json would
        # fail if called with an index that is an array, which happens with
        # compound ids.  Since writing "records" doesn't write the index, we
        # throw it away.
        if predictions.index.dtype == "object":
            predictions = predictions.reset_index(drop=True)
        output = predictions.to_json(orient="records", lines=True, double_precision=3)
    else:
        raise Exception(
            f"Unsupported output format for {fpath}.  Must be .csv, .json, or .jsonl"
        )
    with open(fpath, "wt") as fobj:
        fobj.write(output)


def _eval_array_column(x):
    """Takes a column which is a JSON string and converts it into an array."""
    if type(x) == str:
        a = eval(x)
        if type(a) != list:
            raise Exception(f"Unexpected data in array column: {x}")
        return a
    elif x is None or (type(x) == float and math.isnan(x)):
        return []
    else:
        msg = f"Unexpected data in array column: {x}"
        raise Exception(msg)
