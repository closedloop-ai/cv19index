# IO functions to read and write model and prediction files in an appropriate format.

import json
import logging
import math
import pickle

import pandas as pd
from pandas.io.common import _NA_VALUES

from .util import UserException, schema_dtypes


logger = logging.getLogger(__file__)


def read_model(fpath):
    with open(fpath, "rb") as fobj:
        return pickle.load(fobj)


def read_frame(fpath, schema_path=None, empty_ok=False):
    if schema_path is None:
        schema_path = fpath + ".schema.json"

    with open(schema_path, "rt") as f:
        schema_file = json.load(f)
        idFields = schema_file["idFields"] if "idFields" in schema_file else None
        schema_json = schema_file["schema"]
    names = [x["name"] for x in schema_json]

    if fpath.endswith(".parquet"):
        ret = pd.read_parquet(fpath)
        if (
            schema_file["idFormat"] == "compoundIndex"
            or schema_file["idFormat"] == "compound"
        ):
            ret.index = ret.index.map(_eval_array_column)
        return ret

    dtypes = schema_dtypes(schema_json)

    # Pandas won't read in ints with NA values, so read those in as floats.
    def adj_type(x):
        if x == "int32" or x == "int64":
            return "float64"
        return x

    adj_dtypes = {k: adj_type(x) for k, x in dtypes.items()}

    # Pandas converts a string with "NA" as a real NaN/Null. We don't want this
    # for real string columns. NA can show up as a real flag in customer data and
    # it doesn't mean it should be treated as NaN/Null.
    def na_vals(x):
        # For now, only ignore NA conversion for strings. Structs/etc can still use it.
        if x["dataType"]["dataType"] in ("string"):
            return []
        else:
            return _NA_VALUES

    # needs to be based on schema json, otherwise pandas types are just "object" which
    # tells us little on if we need to retain nan parsing.
    na_values = {x["name"]: na_vals(x) for x in schema_json}

    date_cols = [
        x["name"]
        for x in schema_json
        if x["dataType"]["dataType"] == "date"
        or x["dataType"]["dataType"] == "datetime"
    ]
    # Pandas read_csv ignores the dtype for the index column,
    # so we don't read in with an index column.  Set it later.
    df = pd.read_csv(
        fpath,
        header=0,
        names=names,
        dtype=adj_dtypes,
        parse_dates=date_cols,
        na_values=na_values,
        keep_default_na=False,
    )

    if df.empty and not empty_ok:
        raise UserException(f"The input data is empty and only contains headers.")

    # Convert array columns
    array_cols = [
        x["name"] for x in schema_json if x["dataType"]["dataType"] == "array"
    ]
    for col in array_cols:
        df[col] = df[col].map(_eval_array_column)

    # Now go back and fix all the ints (which we read in as floats to handle NAs)
    df = df.fillna(0)
    for k, x in dtypes.items():
        if x == "int32" or x == "int64":
            df[k] = df[k].astype(x)

    # Sometimes pandas doesn't pay attention to the type we give it.
    # Go back and ask it to really make every column be what we asked.
    for n, t in adj_dtypes.items():
        df[n] = df[n].astype(t)

    # Oh, and if there are date/datetime types that are empty, pandas "helps us out"
    # by converting those to an int(0) instead of None.
    # Thanks pandas, but I'd prefer None here.
    for c in date_cols:
        df.loc[df[c] == 0, [c]] = None

    # Now verify that everything is exactly as it should be.
    incorrect_types = [
        f"{n} expected {t} but was {df[n].dtype}"
        for n, t in adj_dtypes.items()
        if df[n].dtype != t
    ]
    assert len(incorrect_types) == 0, f"Incorrect types:\n" + "\n".join(incorrect_types)

    # Now set the index.
    df = df.set_index(names[0], drop=True)
    return df


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
        logger.error(msg)
        raise Exception(msg)
