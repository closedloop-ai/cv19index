import logging
import math
from datetime import datetime

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


def preprocess_claim(cliam_df: pd.DataFrame, asOfDate: pd.datetime = pd.to_datetime(datetime.now().isoformat()) ):
    asOfPastYear = str(pd.to_datetime(asOfDate) - pd.DateOffset(years=1))

    # limit to last year of claims
    cliam_df = cliam_df[(asOfPastYear <= cliam_df['admitDate']) & (cliam_df['admitDate'] <= asOfDate)]

    # total numbers of days in the ER
    er_visit = cliam_df[cliam_df['erVisit'] == True][['personId', 'admitDate']].groupby(
        'personId').admitDate.nunique().reset_index()
    cliam_df.merge(er_visit, how='left')
    cliam_df['# of ER Visits (12M)'] = cliam_df['admitDate'] + cliam_df['dischargeDate']

    # days from admit to dishcharge
    cliam_df['Inpatient Days'] = cliam_df[cliam_df['inpatient'] == True][['dischargeDate', 'admitDate']].apply(
        lambda x: (pd.to_datetime(x.dischargeDate) - pd.to_datetime(x.admitDate)).days, axis=1)

    # fix na values
    cliam_df = cliam_df.fillna(0)

    # Number of admissions is number of unique admit dates.
    person_df = person_df.rename(columns={'gender': 'Gender', 'admitDate': '# of Admissions (12M)'})

    # Cleaning the diagnosis codoes apply to tall the dx cols
    for column in diagnosis_columns:
        inpatient_df[column] = inpatient_df[column].apply(lambda x: cleanICD10Syntax(str(x)))



def getTestDataFrame(person_df, eligibility_df, inpatient_df, outpatient_df, asOfDate, diagnosis_columns):
    # Getting diagnosis within the past year of asOfDate
    asOfPastYear = str(pd.to_datetime(asOfDate) - pd.DateOffset(years=1))
    inpatient_df = inpatient_df[(asOfPastYear <= inpatient_df['admitDate']) & (inpatient_df['admitDate'] <= asOfDate)]
    outpatient_df = outpatient_df[
        (asOfPastYear <= outpatient_df['serviceDate']) & (outpatient_df['serviceDate'] <= asOfDate)]

    # total numbers of days in the ER
    inpatient_er_visit = inpatient_df[inpatient_df['edAdmit'] == True][['personId', 'admitDate']].groupby(
        'personId').admitDate.nunique().reset_index()
    outpatient_er_visit = outpatient_df[outpatient_df['edVisit'] == True][['personId', 'serviceDate']].groupby(
        'personId').serviceDate.nunique().reset_index()


    # Calculating Age, # of ER Visits, # of Admissions and Inpatient days
    person_df['Age'] = person_df['birthYear'].apply(lambda x: pd.to_datetime('now').year - pd.to_datetime(x).year)
    person_df = person_df.merge(inpatient_er_visit, how='left').merge(outpatient_er_visit, how='left')
    person_df['# of ER Visits (12M)'] = person_df['admitDate'] + person_df['serviceDate']

    # days from admit to dishcharge
    inpatient_df['Inpatient Days'] = inpatient_df[['dischargeDate', 'admitDate']].apply(
        lambda x: (pd.to_datetime(x.dischargeDate) - pd.to_datetime(x.admitDate)).days, axis=1)
    inpaitent_er_days = inpatient_df[['personId', 'Inpatient Days']]
    person_df = person_df.merge(inpaitent_er_days, how='left')

    person_df = person_df.fillna(0)

    person_df = person_df[['personId', '# of ER Visits (12M)', 'gender', 'Age', 'admitDate', 'Inpatient Days']]
    # Number of admissions is number of unique admit dates.
    person_df = person_df.rename(columns={'gender': 'Gender', 'admitDate': '# of Admissions (12M)'})

    # Cleaning the diagnosis codoes apply to tall the dx cols
    for column in diagnosis_columns:
        inpatient_df[column] = inpatient_df[column].apply(lambda x: cleanICD10Syntax(str(x)))
        outpatient_df[column] = outpatient_df[column].apply(lambda x: cleanICD10Syntax(str(x)))

    nodes = pd.read_csv(open('/home/ben/Projects/cv19index/cv19index/resources')))
    edges_df = pd.read_csv(resource_filename('cv19index', 'cv19index/resources/ccsrEdges.txt'))
    edges_df['code'] = edges_df['child'].apply(lambda x: x.split(':')[1])

    # Generating features for each node
    for CCSR, description in nodes.values:
    # Getting the codes
        codes = edges_df[edges_df['parent'].str.contains(CCSR)]
    selected_inpatient = inpatient_df[inpatient_df.isin(codes['code'].values).any(axis=1)]['personId'].values
    selected_outpatient = outpatient_df[outpatient_df.isin(codes['code'].values).any(axis=1)]['personId'].values
    selected_personId = np.unique(np.concatenate((selected_inpatient, selected_outpatient)))

    # Assigning the diagnosis flag to ther person
    description = re.sub("[^\P{P}-/']+", "_", description.replace(")", ""))
    column_name = "Diagnosis of " + description + " in the previous 12 months"
    person_df[column_name] = person_df['personId'].apply(lambda x: True if x in selected_personId else False)

    # Getting the column order for the model
    f = open(resource_filename("cv19index", "resources/xgboost/input.csv.schema.json"))
    column_order = [item['name'] for item in json.load(f)['schema']]
    f.close()

    # returning the needed features.
    return person_df[column_order]