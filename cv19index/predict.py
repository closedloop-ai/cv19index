import argparse
import logging
import math
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from pkg_resources import resource_filename

from .io import read_frame, read_model, write_predictions
from .preprocess import apply_int_mapping
from .shap_top_factors import (
    append_empty_shap_columns,
    calculate_shap_percentile,
    filter_rows_with_index,
    generate_shap_top_factors,
    reset_multiindex,
    select_index,
    shap_score_to_percentile,
)

logger = logging.getLogger(__file__)

PREDICTION_QUANTILES = "prediction_quantiles"


def clean_floats(d: pd.DataFrame):
    new_d = d.copy()
    for k, v in new_d.items():
        if math.isnan(v) or math.isinf(v):
            new_d[k] = None
        elif isinstance(v, np.float64):
            new_d[k] = float(v)
    return new_d


def rescale_predictions(
    predictions: np.ndarray, train_data_stats: Dict[str, Any]
) -> np.ndarray:
    """
    Function to rescale model predictions s.t. they correspond to probabilities.
    This funciton is only called if pos_scale_weight > 1.  Some background can be
    found here: https://github.com/dmlc/xgboost/issues/863

    predictions - the original predictions which need to be rescaled
    train_data_stats - dictionary containing the total number of events and number
        of rare events

    Returns: rescaled predictions which now correspond to probabilities
    """

    # original total number of events
    E = train_data_stats["total_events"]
    # original number of rare events
    B = train_data_stats["rare_events"]
    # original number of non-rare events
    A = E - B
    # new number of non-rare events
    C = A
    # new number of rare events
    D = A
    # new total number of events
    F = C + D

    fac1 = (A / E) / (C / F)
    fac2 = (B / E) / (D / F)

    def rescale_predictions(p):
        a0 = (1 - p) * fac1

        a1 = p * fac2

        return a1 / (a1 + a0)

    predictions = np.apply_along_axis(rescale_predictions, 0, predictions)

    return predictions


def perform_predictions(
    df: pd.DataFrame,
    xmatrix: xgb.DMatrix,
    label: np.ndarray,
    predictor: Dict[str, Any],
    recompute_distribution: bool,
    shap_score_99: float = None,
    shap_cutoff: float = 0.02,
    compute_factors: bool = True,
    factor_cutoff: float = None,
    **kwargs,
) -> Tuple[pd.DataFrame, List[float], float, float]:
    """
    Args:
        compute_factors_at_cutoff: float (0, 1], only compute SHAP for rows
        beyond this cutoff after sorted by prediction
    """
    outcome_column = predictor["outcome_column"]
    mapping = predictor["mapping"]
    model = predictor["model"]
    """
    Build predictions from a trained model, additionally build shap scores and quantiles

    return prediction DataFrame, quantiles, shap_base_value, shap_score_99
    """

    predictions = model.predict(xmatrix)

    # rescale the predictions if model used scale_pos_weight_flag
    if predictor["predictor_type"] == "classification":
        if predictor["hyper_params"]["scale_pos_weight"] > 1:
            logger.info(
                f'Scale pos weight is {predictor["hyper_params"]["scale_pos_weight"]}.'
                " Rescaling predictions to probabilities"
            )
            predictions = rescale_predictions(
                predictions, predictor["train_data_stats"]
            )
        else:
            logger.info(
                f'Scale pos weight is {predictor["hyper_params"]["scale_pos_weight"]}.'
                " No need to rescale predictions"
            )

    # If our index is a list type, convert it to tuples
    index_name = df.index.name
    if df.index.dtype == "object" and df.index.size > 0 and type(df.index[0]) == list:
        df.index = [tuple(l) for l in df.index]
        df.index.name = index_name

    prediction_quantiles = get_quantiles(predictions, predictor, recompute_distribution)
    risk_scores = np.digitize(predictions, prediction_quantiles)

    # need to include the label in the output df for split trains
    # in the case of a composite key DataFrame joining doesnt naively work
    # instead including it at construction time
    pred_dict = {
        df.index.name: df.index,
        "prediction": predictions,
        "risk_score": risk_scores,
    }
    if label is not None:
        pred_dict[outcome_column] = label
    prediction_result = pd.DataFrame(pred_dict, index=df.index)

    # compute top factors and "base value" using SHAP technique
    top_factors, shap_base_value = (None, None)
    if compute_factors:
        if isinstance(factor_cutoff, float):
            compute_factors_cutoff_val = prediction_quantiles[
                int(100 * (1 - factor_cutoff))
            ]
            df_cutoff = df[predictions >= compute_factors_cutoff_val]
            top_factors, shap_base_value = generate_shap_top_factors(
                df_cutoff, model, outcome_column, mapping, **kwargs
            )
            logger.info(
                f"Computed SHAP scores for top {100*factor_cutoff}% of predictions"
                f" resulting in {top_factors.shape[0]} scores."
            )
            prediction_result = prediction_result.join(top_factors)
            empty_list = np.empty(shape=(0,))
            for col_name in top_factors.columns:
                prediction_result[col_name] = prediction_result[col_name].apply(
                    lambda d: d
                    if (isinstance(d, list) or isinstance(d, np.ndarray))
                    else empty_list
                )
        else:
            top_factors, shap_base_value = generate_shap_top_factors(
                df, model, outcome_column, mapping, **kwargs
            )
            prediction_result = prediction_result.join(top_factors)

    # If index is tuples, convert it back to list
    if type(prediction_result.index[0]) == tuple:
        prediction_result[index_name] = [list(l) for l in prediction_result.index]
        prediction_result = prediction_result.set_index(index_name, drop=False)

    if compute_factors:
        # Calculate 99th percentile shap score if it doesn't already exist
        if shap_score_99 is None:
            abs_shap_scores = np.abs(
                np.concatenate(
                    prediction_result[
                        ["pos_shap_scores", "neg_shap_scores"]
                    ].values.flatten()
                )
            )
            if len(abs_shap_scores) < 1:
                shap_score_99 = 0.0
            else:
                shap_score_99 = np.percentile(abs_shap_scores, 99)

        logger.info(f"Cutoff for SHAP values: {shap_cutoff}")

        prediction_result["neg_index_filter"] = prediction_result[
            "neg_shap_scores"
        ].apply(filter_rows_with_index, args=(shap_cutoff,))
        prediction_result["pos_index_filter"] = prediction_result[
            "pos_shap_scores"
        ].apply(filter_rows_with_index, args=(shap_cutoff,))

        prediction_result["neg_shap_scores_w"] = (
            prediction_result["neg_shap_scores"] / shap_score_99
        )
        prediction_result["pos_shap_scores_w"] = (
            prediction_result["pos_shap_scores"] / shap_score_99
        )

        prediction_result["neg_shap_scores"] = (
            reset_multiindex(prediction_result)[["neg_shap_scores", "neg_index_filter"]]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )
        prediction_result["neg_shap_scores_w"] = (
            reset_multiindex(prediction_result)[
                ["neg_shap_scores_w", "neg_index_filter"]
            ]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )
        prediction_result["neg_factors"] = (
            reset_multiindex(prediction_result)[["neg_factors", "neg_index_filter"]]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )
        prediction_result["neg_patient_values"] = (
            reset_multiindex(prediction_result)[
                ["neg_patient_values", "neg_index_filter"]
            ]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )

        prediction_result["pos_shap_scores"] = (
            reset_multiindex(prediction_result)[["pos_shap_scores", "pos_index_filter"]]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )
        prediction_result["pos_shap_scores_w"] = (
            reset_multiindex(prediction_result)[
                ["pos_shap_scores_w", "pos_index_filter"]
            ]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )
        prediction_result["pos_factors"] = (
            reset_multiindex(prediction_result)[["pos_factors", "pos_index_filter"]]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )
        prediction_result["pos_patient_values"] = (
            reset_multiindex(prediction_result)[
                ["pos_patient_values", "pos_index_filter"]
            ]
            .apply(lambda x: select_index(*x), axis=1)
            .values
        )

        prediction_result = prediction_result.drop(
            columns=["neg_index_filter", "pos_index_filter"]
        )
    else:
        prediction_result = append_empty_shap_columns(prediction_result)

    return prediction_result, prediction_quantiles, shap_base_value, shap_score_99


def get_quantiles(
    predictions: np.ndarray, predictor: Dict[str, Any], recompute_distribution: bool
) -> List[float]:
    """
    Helper to get the quantiles of a prediction DataFrame
    """
    if recompute_distribution:
        q = np.array(range(100))
        prediction_quantiles = list(np.percentile(predictions, q=q))
    else:
        prediction_quantiles = predictor[PREDICTION_QUANTILES]
    return sorted(prediction_quantiles)


def get_agg_preds(val_to_preds: Dict[str, Any]) -> pd.DataFrame:
    """
    Take the dictionary that maps the split key to the model output
    aggregate the prediction DataFrames together as well as join the label data
    which can then be used to build test results

    """
    agg_preds = None
    for val, results in val_to_preds.items():
        if agg_preds is None:
            agg_preds = results["predictions"]
        else:
            agg_preds = pd.concat((agg_preds, results["predictions"]))

    return agg_preds


def run_model(run_df: pd.DataFrame, predictor: Dict, **kwargs) -> pd.DataFrame:
    df_inputs = apply_int_mapping(
        predictor["mapping"], run_df, error_unknown_values=False
    )

    df_inputs = reorder_inputs(df_inputs, predictor)
    run = xgb.DMatrix(df_inputs)
    factor_cutoff = (
        kwargs["predict_factor_cutoff"] if "predict_factor_cutoff" in kwargs else 1.0
    )
    predictions, prediction_quantiles, shap_base_value, _ = perform_predictions(
        df_inputs,
        run,
        None,
        predictor,
        recompute_distribution=False,
        shap_score_99=predictor["shap_score_99"],
        factor_cutoff=factor_cutoff,
        **kwargs,
    )

    shap_pct = predictor.get("shap_pct")
    if shap_pct is None:
        shap_pct = calculate_shap_percentile(predictions)

    predictions["pos_shap_percentile"] = predictions["pos_shap_scores"].apply(
        shap_score_to_percentile, args=(shap_pct,)
    )
    predictions["neg_shap_percentile"] = predictions["neg_shap_scores"].apply(
        shap_score_to_percentile, args=(shap_pct,)
    )

    return predictions


def reorder_inputs(df_inputs: pd.DataFrame, predictor: Dict[str, Any]) -> pd.DataFrame:
    """This code reorders in the input data frame columns to match the expected order
    from training. If the lists of columns don't match, no error is raised, as this
    will cause issues later."""
    if set(predictor["model"].feature_names) == set(df_inputs.columns) and predictor[
        "model"
    ].feature_names != list(df_inputs.columns):
        logger.info("Reordering test inputs to match training.")
        return df_inputs.loc[:, predictor["model"].feature_names]
    return df_inputs


def do_run(
    input_fpath: str, schema_fpath: str, model_fpath: str, output_fpath: str, **kwargs
) -> None:
    """
    Read in a trained model, creating predictions for the data located at the input path
    writes out predictions.
    """
    logger.info(f"Running {model_fpath} using {input_fpath} to {output_fpath}")
    run_df = read_frame(input_fpath, schema_fpath, empty_ok=True)

    if not run_df.empty:
        model = read_model(model_fpath)

        out = run_model(run_df, model, **kwargs)
        write_predictions(out, output_fpath)
    else:
        #: If there are no inputs to run predictions on, bypass
        #: executing the model and output an successful empty file.
        write_predictions(run_df, output_fpath)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-m", "--model", choices=["xgboost"], default="xgboost")

    args = parser.parse_args()

    if args.model == "xgboost":
        model = resource_filename("cv19index", "resources/xgboost/model.pickle")
        schema = resource_filename(
            "cv19index", "resources/xgboost/input.csv.schema.json"
        )

    do_run(args.input, schema, model, args.output)
