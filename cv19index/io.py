# IO functions to read and write model and prediction files in an appropriate format.

import json
import logging
import math
import pickle
from typing import Dict
import pandas as pd
from pandas.io.common import _NA_VALUES
from pkg_resources import resource_filename

from .util import schema_dtypes

logger = logging.getLogger(__name__)

INDEX = 'personId'


def read_model(fpath):
    with open(fpath, "rb") as fobj:
        return pickle.load(fobj)


def get_na_values(dtypes):
    # Pandas converts a string with "NA" as a real NaN/Null. We don't want this
    # for real string columns. NA can show up as a real flag in data and
    # it doesn't mean it should be treated as NaN/Null.
    def na_vals(x):
        # For now, only ignore NA conversion for strings. Structs/etc can still use it.
        if x in ("string", ):
            return []
        else:
            return _NA_VALUES

    return {k: na_vals(v) for k, v in dtypes.items()}


def read_claim(fpath: str) -> pd.DataFrame:
    schame_fpath = resource_filename("cv19index", "resources/claims.schema.json")
    return read_frame(fpath, schame_fpath, date_cols = ['admitDate', 'dischargeDate'])

import collections

def read_demographics(fpath: str) -> pd.DataFrame:
    schema_fpath = resource_filename("cv19index", "resources/demographics.schema.json")
    df = read_frame(fpath, schema_fpath).set_index('personId')
    duplicates = [x[0] for x in collections.Counter(df.index.values).items() if x[1] > 1]
    assert len(duplicates) == 0, f"Duplicate person ids in demographics file: {duplicates[:5]}"
    return df


def read_frame(fpath, schema_fpath, date_cols = []) -> pd.DataFrame:
    XLS = ('xls', 'xlsx', 'xlsm', 'xlsb', 'odf')

    def is_excel(fpath: str) -> bool:
        return fpath.endswith(XLS)

    with open(schema_fpath) as f:
        schema = json.load(f)
    dtypes = schema_dtypes(schema["schema"])

    if is_excel(fpath):
        df = read_excel(fpath, dtypes, date_cols)
    elif fpath.endswith(".parquet"):
        df = read_parquet(fpath, dtypes, date_cols)
    elif fpath.endswith(".csv"):
        df = read_csv(fpath, dtypes, date_cols)
    else:
        raise TypeError(f'This script reads files based the extension.\n'
                        f'The known extensions are {", ".join(XLS)}, .parquet, .csv'
                        f'Please ensure your file is one of those file types with correct file extension.')

    return df


def read_excel(fpath, dtype, date_cols) -> pd.DataFrame:
    date_cols = [k for k, v in dtype.items() if v == "date"]
    na_values = get_na_values(dtype)

    df = pd.read_excel(
        fpath,
        header=0,
        dtype=dtype,
        parse_dates=date_cols,
        na_values=na_values,
        keep_default_na=False
    )

    return df


def read_parquet(fpath, dtype, date_cols) -> pd.DataFrame:
    df = pd.read_parquet(fpath)
    # Now set the index.
    df = df.set_index(INDEX)
    return df


def read_csv(fpath: str, dtype: Dict, date_cols) -> pd.DataFrame:
    na_values = get_na_values(dtype)

    # Pandas won't read in ints with NA values, so read those in as floats.
    def adj_type(x):
        if x == "int32" or x == "int64":
            return "float64"
        return x

    df = pd.read_csv(
        fpath,
        header=0,
        na_values=na_values,
        index_col=False,
        keep_default_na=False,
    )

    # Oh, and if there are date/datetime types that are empty, pandas "helps us out"
    # by converting those to an int(0) instead of None.
    # Thanks pandas, but I'd prefer None here.
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], infer_datetime_format = True)

    return df


def write_predictions(predictions, fpath):
    if fpath.endswith(".csv"):
        output = predictions.to_csv(index=False, float_format="%f")
    elif fpath.endswith(".json"):
        if (
            predictions.index.dtypes == "object"
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
        if predictions.index.dtypes == "object":
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
