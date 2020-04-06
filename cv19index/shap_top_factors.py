import logging
import operator

import numpy as np
import pandas as pd
import shap

__all__ = [
    "append_empty_shap_columns",
    "calculate_shap_percentile",
    "filter_rows_with_index",
    "generate_shap_top_factors",
    "reset_multiindex",
    "select_index",
    "shap_score_to_percentile",
]

logger = logging.getLogger(__file__)
MAX_FEATURES = 10


def get_shap_factor_values(row, shap_score_dict, key):
    if key == "pos":
        reverse = True
    else:
        reverse = False

    sorted_shap_score_dict = sorted(
        shap_score_dict.items(), key=operator.itemgetter(1), reverse=reverse
    )

    # loop thru top factors and extract factor name, its shap value, and the patient's
    # value for that factor
    num_features = min(len(sorted_shap_score_dict), MAX_FEATURES)

    factors = [sorted_shap_score_dict[idx][0][:-11] for idx in range(num_features)]
    shap_scores = [
        round(row[sorted_shap_score_dict[idx][0]], 3) for idx in range(num_features)
    ]
    patient_values = [
        row[sorted_shap_score_dict[idx][0][:-11]] for idx in range(num_features)
    ]

    row[key + "_factors"] = factors
    row[key + "_shap_scores"] = np.array(shap_scores)
    row[key + "_patient_values"] = patient_values

    return row


# function to construct top factor df in a vectorized way
def build_top_factor_df(row, shap_base_value):
    # build dictionary for positive and negative shap_scores
    pos_shap_score_dict = {}
    neg_shap_score_dict = {}

    for k, v in row.to_dict().items():
        if v is not None and k.endswith("_shap_score") and v > 0.0:
            pos_shap_score_dict[k] = v
        elif v is not None and k.endswith("_shap_score") and v < 0.0:
            neg_shap_score_dict[k] = v

    row = get_shap_factor_values(row, pos_shap_score_dict, "pos")
    row = get_shap_factor_values(row, neg_shap_score_dict, "neg")

    return row


def unmap_int_cols(df, outcome_column, mapping):
    """
    function to unmap categorical columns

    df - dataframe
    mapping - mapping dictionary

    returns: df with categorical columns mapped back to original values
    """
    for key in mapping.keys():

        # build the unmapping dictionary for "key"
        unmap_dict = {v: k for k, v in mapping[key].items()}

        # apply the unmapping to the items in column "key"
        if key != outcome_column:
            # df[key] = df[key].apply(lambda x: unmap_dict[x])
            df[key] = df[key].apply(
                lambda x: unmap_dict[x] if (np.all(pd.notnull(x))) else None
            )

    return df


def append_empty_shap_columns(df):
    factor_column_list = [
        "pos_factors",
        "pos_shap_scores",
        "pos_patient_values",
        "pos_shap_scores_w",
        "pos_shap_percentile" "neg_factors",
        "neg_shap_scores",
        "neg_patient_values",
        "neg_shap_scores_w",
        "neg_shap_percentile",
    ]
    # Make a column containing an empty list for every row.
    l = []
    v = df[df.columns[0]].apply(lambda x: l)
    for c in factor_column_list:
        df[c] = v
    return df


def generate_shap_top_factors(
    df, model, outcome_column, mapping, shap_approximate=False, **kwargs
):
    """Computes the contributing factors using the SHAP package and orders 
    them by absolute value

    Input: feature dataframe and the xgboost model object
    Output: dataframe of ordered contributing factors,
            their shap value and std devs from population mean
            and the patient's feature value
    """
    # compute the shap values
    logger.warning(f"Computing SHAP scores.  Approximate = {shap_approximate}")
    shap_values = shap.TreeExplainer(model).shap_values(
        df, approximate=shap_approximate, check_additivity=False
    )
    logger.warning(f"SHAP values completed")
    shap_df = pd.DataFrame(shap_values)
    #logger.info(f"SHAP: {shap_df.shape[0]} {df.shape[0]}")

    shap_base_value = shap_df.iloc[0, -1]

    # drop the bias term (last column) - unless it's not present.
    if shap_df.shape[1] == df.shape[1] + 1:
        shap_df.drop(shap_df.shape[1] - 1, axis=1, inplace=True)

    assert shap_df.shape[0] == df.shape[0], (
        f"shap_values was {shap_df.shape}, didn't have the same number of rows"
        f" as {df.shape}"
    )
    assert shap_df.shape[1] == df.shape[1], (
        f"shap_values was {shap_df.shape}, did not have the same number of columns"
        f" as {df.shape} {shap_df.columns.tolist()}"
    )

    # make a dict for feature name
    df_columns = df.columns.tolist()
    shap_df_columns = shap_df.columns.tolist()
    feature_map = {
        shap_df_columns[i]: df_columns[i] + "_shap_score"
        for i in range(len(df_columns))
    }

    # rename columns to align with passed in df
    shap_df.rename(columns=feature_map, inplace=True)
    shap_df.index = df.index

    # join original data with shap_df
    assert shap_df.shape[0] == df.shape[0]
    assert len(set(shap_df.index.values)) == df.shape[0]
    assert len(set(df.index.values)) == df.shape[0]
    assert (shap_df.index.values == df.index.values).all()
    #joined_df = pd.merge(df, shap_df, left_index=True, right_index=True)
    joined_df = df.join(shap_df)
    assert joined_df.shape[0] == df.shape[0], f"{joined_df.shape[0]} == {df.shape[0]}"

    # unmap categorical columns
    joined_df = unmap_int_cols(joined_df, outcome_column, mapping)

    # initialize top factor columns (capped at MAX_FEATURES)
    num_features = len(shap_df_columns)
    if num_features > MAX_FEATURES:
        num_features = MAX_FEATURES
    factor_column_list = [
        "pos_factors",
        "pos_shap_scores",
        "pos_patient_values",
        "neg_factors",
        "neg_shap_scores",
        "neg_patient_values",
    ]

    # use build_top_factor_df function to compute top factor info
    joined_df = joined_df.apply(build_top_factor_df, args=(shap_base_value,), axis=1)

    # keep only the top factor columns
    top_factor_df = joined_df[factor_column_list]

    return top_factor_df, shap_base_value


def calculate_shap_percentile(pred):
    """Computes the percentile of a distribution of SHAP scores
        input: dataframe with pos and neg shap score column_names
        output: array of percentiles for abs() value of all shap scores
    """

    pos_shap_scores = np.abs(np.hstack(pred["pos_shap_scores"]))
    neg_shap_scores = np.abs(np.hstack(pred["neg_shap_scores"]))
    all_shap_scores = np.concatenate([pos_shap_scores, neg_shap_scores])

    q = np.array(range(100))

    if len(all_shap_scores) == 0:
        all_shap_scores = [0.0]
    shap_pct = np.percentile(all_shap_scores, q)

    return shap_pct


def filter_rows_with_index(r, cutoff):
    """Return index of items in the list above the cutoff"""
    return [i for i, x in enumerate(r) if abs(x) > cutoff]


def select_index(r, i):
    """Return the items in the list from the specified index list"""
    return list(np.take(r, i))


def reset_multiindex(df):
    return df.reset_index(drop=True, level=0)
