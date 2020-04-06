from cv19index.io import read_demographics, read_claim
from .predict import do_run_claims, do_run
from .preprocess import preprocess_xgboost
from datetime import datetime
import pandas as pd


def test_do_run_claims():
    do_run_claims('examples/demographics.csv',
                  'examples/claims.xlsx',
                  f'{datetime.now().strftime("%Y-%M-%dT%H:%m:%S")}-prediction_summary.csv',
                  'xgboost',
                  pd.to_datetime('2018-12-01'))


def test_do_run():
    asOfDate = pd.to_datetime('2018-12-01')
    demo_df = read_demographics('examples/demographics.csv')
    claim_df = read_claim('examples/claims.xls')

    result_df = preprocess_xgboost(claim_df, demo_df, asOfDate)
    result_df = result_df.set_index('personId', drop=True)

    result_name = "cv19index/resources/xgboost/example_input.csv"
    result_df.to_csv(result_name, index=False, float_format="%f")

    do_run(result_name,
           "cv19index/resources/xgboost/input.csv.schema.json",
           "cv19index/resources/xgboost/model.pickle",
           "cv19index/resources/xgboost/output.csv")

