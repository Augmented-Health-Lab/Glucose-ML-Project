# 3_Glucose-ML-collection - Standardized CGM Files & Participant Metadata

This directory contains the standardized Continuous Glucose Monitoring (CGM) data and Metadata for open-access Glucose-ML projects.

---
## What’s included?
The Glucose-ML team has taken the liberty to pre-standardize all datasets for your convenience. Additionally, you will find a metadata table describing the avalible metadata for each participant with valid CGM data. Pre-standardized data and metadata are probvided only for open-access datasets. A README containing dataset access information can be found for controlled-access datasets.

Directory contents by dataset avalibility:

1. Open-access Datasets:
    * `{Dataset}/Extracted-glucose-files`
    * `{Dataset}/{Dataset}_metadata.csv`

2. Controlled-access Datasets:
    * `{Dataset}/README.md`

---
## Data Dictionary
This section provides descriptive information for each data column found in the open-access dataset directories. For detailed methods, please refer to the Glucose-ML publication.

1. `{Dataset}/Extracted-glucose-files`:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `{Dataset}/{Dataset}_metadata.csv`
    - *person_id*: Unique participant identifier.
    - *diabetes_type*: Diabetes status as reported by the data curator. If not reported, this value is infered by hba1c_% if available.
    - _age_: Age of the participant as reported by the data curator.
    - _gender_: Gender/Sex of the participant as reported by the data curator.
    - *race_ethnicity*: Race/Ethnicity of the participant as reported by the data curator.
    - _hba1c_%: Hemoglobin A1C percentage of the participant as reported by the data curator.
    - *CGM_type*: Continuous glucose monitoring (CGM) device used for data collection as reported by the data curator.
    - *glucose_level_record_count*: Total number of Glucose-ML standardized CGM records for the participant
    - *average_glucose_level_mg_dl*: Mean glucose level (mg/dL) across all Glucose-ML standardized readings for the participant.
    - *count_days_with_CGM_data*: Total number of unique days on which at least one glucose reading was recorded.




<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal_white.svg" alt="Glucose-ML logo" width="450">
</p>