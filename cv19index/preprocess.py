import logging
import math

import numpy as np
import pandas as pd

from .util import none_or_nan, nonnull_column

logger = logging.getLogger(__name__)


def apply_int_mapping(mapping, data, error_unknown_values=True):
    """Maps strings to integers in a Data Frame using a prespecified mapping.
    Takes a mapping of values and applies it to the data.
    Null or None values will be mapped over.
    Params:
        mapping - A nested dictionary of {colName: {'value':mapped_value}}
        data - The DataFrame to map
    Returns:
        A new DataFrame with the string columns mapped.
    """
    ret = data.loc[:, list(set(data.columns) - set(mapping.keys()))].copy()
    for col, value_map in mapping.items():
        if col not in data.columns:
            if error_unknown_values:
                logger.error(f'No column named "{col}" in {data.columns}')
                raise ValueError(f'No column named "{col}" in {data.columns}')
            else:
                continue

        ret[col] = data[col].map(value_map)

        # Check if any non-null values in the original array are not mapped
        if error_unknown_values and np.any(
            np.logical_and(np.isnan(ret[col]), nonnull_column(data[col]))
        ):
            raise ValueError(
                f"Column '{col}' had invalid values:"
                f" {set(data[col].unique()) - set(value_map.keys())}."
                f"  Valid values are {set(value_map.keys())}"
            )
    return ret
