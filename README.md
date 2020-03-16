# The COVID-19 Vulnerability Index (CV19 Index)

This repository contains the source code, models, and example usage of the COVID-19 Vulnerability Index (CV19 Index).  The CV19 Index is a predictive model that identifies people who are likely to have a heightened vulnerability to severe complications from COVID-19 (commonly referred to as “The Coronavirus”).  The CV19 Index is intended to help hospitals, federal / state / local public health agencies and other healthcare organizations in their work to identify, plan for, respond to, and reduce the impact of COVID-19 in their communities.

Full information on the CV19 Index, including the links to a full FAQ, User Forums, and information about upcoming Webinars is available at http://cv19index.com

This repository provides information for those interested in computing COVID-19 Vulnerability Index scores.

## Versions of the CV19 Index

There are 3 different versions of the CV19 Index.  Each is a different predictive model for the CV19 Index.  The models represent different tradeoffs between ease of implementation and overall accuracy.  A full description of the creation of these models is available in the accompanying paper, "Building a COVID-19 Vulnerability Index" (http://cv19index.com).

The 3 models are:

* _Simple Linear_ - A simple linear logistic regression model that uses only 14 variables.  An implementation of this model is included in this package.  This model had a 0.731 ROC AUC on our test set.

* _Open Source ML_ - An XGBoost model, packaged with this repository, that uses Age, Gender, and 500+ features defined from the [CCSR](https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp)  categorization of diagnosis codes.  This model had a 0.811 ROC AUC on our test set.

* _Free Full_ - An XGBoost model that fully utilizes all the data available in Medicare claims, along with geographically linked public and Social Determinants of Health data.  This model provides the highest accuracy of the 3 CV19 Indexes but requires additional linked data and transformations that preclude a straightforward open-source implementation.  ClosedLoop is making a free, hosted version of this model available to healthcare organizations.  For more information, see http://cv19index.com.

### Model Performance
We evaluate the model using a full train/test split.  The models are tested on 369,865 individuals.  We express model performance using the standard ROC curves, as well as the following metrics:
<table style="width:100%">
  <tr>
    <th>Model</th>
    <th>ROC AUC</th>
    <th>Sensitivity as 3% Alert Rate</th>
    <th>Sensitivity as 5% Alert Rate</th>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>.731</td>
    <td>.214</td>
    <td>.314</td>
  </tr>
  <tr>
    <td>XGBoost, Diagnosis History + Age</td>
    <td>.810</td>
    <td>.234</td>
    <td>.324</td>
  </tr>
  <tr>
    <td>XGBoost, Full Features</td>
    <td>.810</td>
    <td>.251</td>
    <td>.336</td>
  </tr>
</table>

<img src="./img/roc.png" />


## Computing the CV19 Index for a patient population

This section describes how to run the _Simple Linear_ and _Open Source ML_ versions of the CV19 Index using this package.  The models work off of CSV files or Pandas data frames.

#### Input Data

The input to the models is a CSV file or Pandas data frame, where each row represents a person and the columns are different computed features, or attributes, for each person.  To use the model you need to calculate all features appropriate for that model.    See the Data Preparation section below for an example on how to prepare data.

#### Output Format

See the example [Jupyter notebook](examples/Tutorial.ipynb) for a description of the output data returned.

### Data Preparation

An [Jupyter notebook](examples/Tutorial.ipynb) with a worked example showing how to prep a data set is provided in the [examples](examples) folder.  This folder begins with a sample set of claims data and performs the necessary transformations to generate the input to the model.

### PyPI Install

```bash
pip install cv19index
```

### Executing Model

To execute the xgboost model.

```bash
cv19index input.csv output.csv
```

Using the provided example data files.

```bash
cv19index examples/xgboost/example_input.csv examples/xgboost/example_prediction.csv
```

Using from within python using a pandas dataframe as input and get the predictions out as a pandas dataframe.

```python
from pkg_resources import resource_filename
model_fpath = resource_filename("cv19index", "resources/xgboost/model.pickle")
model = read_model(model_fpath)
predictions_df = run_model(input_df, model)
```
