from cv19index.predict import do_run_claims
from datetime import datetime
import pandas as pd

do_run_claims('examples/data/demographics.csv',
              'examples/data/claims.xls',
              "cv19index/resources/xgboost/model.pickle",
              f'{datetime.now().strftime("%Y-%M-%dT%H:%m:%S")}-prediction_summary.csv',
              'xgboost',
              pd.to_datetime('2018-12-01'))

