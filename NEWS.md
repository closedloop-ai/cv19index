CV19 Index Change Log
==================

This file records the changes in CV19 Index library in reverse chronological order.

## v1.1.4 (2020.05.12)

* Fixes #21, #25, #26, #28 - handle  cases where there is no inpatient data.  Also fixed some install issues related to having different versions of various dependencies.

## v1.1.3 (2020.04.07)

* Fixes #25 - resources directory missing with PIP install

## v1.1.2 (2020.04.07)

* Fixes #21 - error with newer versions of Pandas

## v1.1.1 (2020.04.06)

* Fixes handling of Excel files for input
* Fixes some broken links in the README.

## v1.1.0 (2020.04.06)
This release is a significant update 

#### All ages model

This release incorporates a new model that is now appropriate for adults ages 18 and over.  This model was trained on a combination of the original CMS Medicare data along with additional data provided by [HealthFirst](https://healthfirst.org/).  

The original Medicare only 'xgboost' model is still available by adding a `-m xgboost` option to cv19index.  However, even for Medicare populations we recommend moving to the new `xgboost_all_ages` model.  This model is now the default.  

#### Other updates

* The README has been completely rewritten to make the usage more clear.
* We have added prebuilt whl files to make instlalation on windows easier.
* Documented the input and output formats more clearly.
* Removed the old preprocessing code from version 1.0.0
* Corrected a bug with the "# of Admissions" feature.
* Corrected a bug where 3 character ICD-10 codes would not be mapped to CCSR.
* Added a "features.csv" option to enable users to see the result of preprocessing. 
* Added a "run_cv19index.py" script that neables running the package without installing from PyPI.
* Several asserts have been added to verify that data types and row counts are correct through the code.

##### Acknowledgements
Many thanks to [HealthFirst](https://healthfirst.org/) for being one of the first users of the model and for allowing us to use their data in order to create a model for all ages. 

<img src=https://healthfirst.org/wp-content/themes/healthfirst2019/assets/images/HealthfirstColorLogo-Tag.png width=300/>  


## v1.0.2 (2020.03.28)

Packaged preprocessing code and maint he main entry point `do_run_claims` instead of `do_run`
 

## v1.0.1 (2020.03.16)

Initial Release
