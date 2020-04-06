import json
import logging
import math
import regex as re
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
    code = str(code)
    if len(code) > 3 and '.' not in code:
        return code[:3] + '.' + code[3:]
    else:
        return code


def preprocess_xgboost(claim_df: pd.DataFrame, demo_df: pd.DataFrame, asOfDate: pd.datetime):
    DIAGNOSIS_COLS = ['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13',
                      'dx14', 'dx15', 'dx16']
    preprocessed_df = demo_df.loc[:,['gender', 'age']].rename(columns={'gender': 'Gender', 'age': 'Age'})

    logger.debug(f"Beginning claims data frame preprocessing, raw data frame as follows.")
    logger.debug(claim_df.head(5))

    asOfPastYear = str(pd.to_datetime(asOfDate) - pd.DateOffset(years=1))[:10]

    # limit to last year of claims
    orig_len = claim_df.shape[0]
    claim_df = claim_df[(asOfPastYear <= claim_df['admitDate'])
                        & (claim_df['admitDate'] <= asOfDate)]
    logger.debug(f"Filtered claims to just those within the dates {asOfPastYear} to {asOfDate}.  Claim count went from {orig_len} to {len(claim_df)}")

    # total numbers of days in the ER
    er_visit = claim_df.loc[claim_df['erVisit'] == True, :].groupby('personId').admitDate.nunique()
    preprocessed_df['# of ER Visits (12M)'] = er_visit

    inpatient_rows = claim_df.loc[claim_df['inpatient'] == True, :]
    preprocessed_df['# of Admissions (12M)'] = inpatient_rows.groupby('personId').admitDate.nunique()
    inpatient_days = pd.Series((inpatient_rows['dischargeDate'].dt.date - inpatient_rows['admitDate'].dt.date).dt.days, index=claim_df['personId'])
    preprocessed_df['Inpatient Days'] = inpatient_days.groupby('personId').sum()

    # Cleaning the diagnosis codes to apply to all the dx cols
    logger.debug(f"Cleaning diagnosis codes.")
    used_diags = [x for x in DIAGNOSIS_COLS if x in claim_df.columns]
    for column in used_diags:
        claim_df[column] = claim_df.loc[:,column].map(cleanICD10Syntax, na_action='ignore')

    # Generating features for each node
    logger.debug(f"Computing diagnosis flags.")
    nodes = pd.read_csv(resource_filename('cv19index', 'resources/ccsrNodes.txt'))
    edges_df = pd.read_csv(resource_filename('cv19index', 'resources/ccsrEdges.txt'))
    edges_df['code'] = edges_df['child'].apply(lambda x: x.split(':')[1])

    for CCSR, description in nodes.values:
        # Getting the codes
        codes = set(edges_df[edges_df['parent'].str.contains(CCSR)]['code'].values)
        #logger.debug(f"Codes are {codes}")
        matches = claim_df.loc[:,used_diags].isin(codes).any(axis=1)
        #logger.debug(f"Matches are {matches.shape[0]}\n{matches.head()}")
        selected_claim = claim_df.loc[matches, 'personId']
        #logger.debug(f"selected_claim are {selected_claim.shape[0]}\n{selected_claim.head()}")
        selected_personId = np.unique(selected_claim)
        #logger.debug(f"Selected are {selected_personId.shape[0]}")

        # Assigning the diagnosis flag to the person
        description = re.sub("[^\P{P}-/']+", "_", description.replace(")", ""))
        column_name = "Diagnosis of " + description + " in the previous 12 months"
        preprocessed_df[column_name] = pd.Series(True, index=selected_personId, dtype=np.bool)
        preprocessed_df[column_name] = preprocessed_df[column_name].fillna(False)
        #preprocessed_df[column_name] = preprocessed_df[column_name].astype(np.bool)
        #logger.debug(f"Final is\n{preprocessed_df[column_name]}")

    # fill in 0's
    preprocessed_df.fillna(0, inplace=True)

    logger.debug(f"Preprocessing complete.")
    return preprocessed_df
