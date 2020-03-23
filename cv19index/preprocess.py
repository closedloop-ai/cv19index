import json
import logging
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

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


def cleanICD10Syntax(code):
    if len(code) > 3 and '.' not in code:
        return code[:3] + '.' + code[3:]
    else:
        code


def preprocess_claim(claim_df: pd.DataFrame, asOfDate: pd.datetime = pd.to_datetime('2018-06-01')):
    DIAGNOSIS_COLS = ['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13',
                      'dx14', 'dx15', 'dxE1']

    logger.info(f"Beginning claim data frame preprocessing, raw data frame as follows.")
    logger.info(claim_df.head(5))
    logger.info(claim_df.dtypes)
    logger.info(claim_df.index)

    asOfPastYear = str(pd.to_datetime(asOfDate) - pd.DateOffset(years=1))

    # limit to last year of claims
    claim_df = claim_df[(asOfPastYear <= pd.to_datetime(claim_df['admitDate']))
                        & (pd.to_datetime(claim_df['admitDate']) <= asOfDate)]

    # total numbers of days in the ER
    er_visit = claim_df[claim_df['erVisit'] == True][['personId', 'admitDate']].groupby(
        'personId').admitDate.nunique().reset_index()
    er_visit = er_visit.rename(columns={'admitDate': '# of Admissions (12M)'})
    claim_df = pd.merge(claim_df, er_visit, left_on='personId', right_on='personId', how='left')
    claim_df['# of ER Visits (12M)'] = claim_df['# of Admissions (12M)']

    # days from admit to discharge
    claim_df['Inpatient Days'] = claim_df[['dischargeDate', 'admitDate']].apply(
        lambda x: (pd.to_datetime(x.dischargeDate) - pd.to_datetime(x.admitDate)).days, axis=1)

    # fix na values
    claim_df = claim_df.fillna(0)

    # rename as needed
    claim_df = claim_df.rename(columns={'gender': 'Gender'})

    # Cleaning the diagnosis codoes apply to tall the dx cols
    for column in DIAGNOSIS_COLS:
        claim_df[column] = claim_df[column].apply(lambda x: cleanICD10Syntax(str(x)))

    nodes = pd.read_csv(resource_filename('cv19index', 'resources/ccsrNodes.txt'))
    edges_df = pd.read_csv(resource_filename('cv19index', 'resources/ccsrEdges.txt'))
    edges_df['code'] = edges_df['child'].apply(lambda x: x.split(':')[1])

    # Generating features for each node
    for CCSR, description in nodes.values:
        # Getting the codes
        codes = edges_df[edges_df['parent'].str.contains(CCSR)]
        selected_claim = claim_df[claim_df.isin(codes['code'].values).any(axis=1)]['personId'].values
        selected_personId = np.unique(np.concatenate(selected_claim))

        # Assigning the diagnosis flag to ther person
        description = re.sub("[^\P{P}-/']+", "_", description.replace(")", ""))
        column_name = "Diagnosis of " + description + " in the previous 12 months"
        claim_df[column_name] = claim_df['personId'].apply(lambda x: True if x in selected_personId else False)

    # Getting the column order for the model
    with open(resource_filename("cv19index", "resources/xgboost/input.csv.schema.json")) as f:
        column_order = [item['name'] for item in json.load(f)['schema']]

    # returning the needed features.
    result = claim_df[column_order]

    logger.info(f"Preprocessing complete data frame as follows.")
    logger.info(result.head(5))
    logger.info(result.dtype)
