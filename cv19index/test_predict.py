from cv19index.io import read_demographics, read_claim
from .predict import do_run_claims, do_run
from .preprocess import preprocess_xgboost
from datetime import datetime
import pandas as pd


def test_do_run_claims():
    do_run_claims('examples/demographics.csv',
                  'examples/claims.csv',
                  f'predictions-{datetime.now().strftime("%Y-%M-%dT%H-%m-%S")}.csv',
                  'xgboost_all_ages',
                  pd.to_datetime('2018-12-01'))

def test_do_run_claims_xlsx():
    do_run_claims('examples/demographics.xlsx',
                  'examples/claims.xlsx',
                  f'predictions-{datetime.now().strftime("%Y-%M-%dT%H-%m-%S")}.csv',
                  'xgboost_all_ages',
                  pd.to_datetime('2018-12-01'))
