# D1NAMO
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *D1NAMO* dataset.

## Contents

1. `D1NAMO-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `D1NAMO_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
    - *person_id*: Unique participant identifier.
    - *diabetes_type*: Diabetes status as reported by the original dataset creators. If not reported, this value is inferred using hba1c_% if available.
    - _age_: Age of the participant by the original dataset creators.
    - _gender_: Gender/Sex of the participant by the original dataset creators.
    - *race_ethnicity*: Race/Ethnicity of the participant by the original dataset creators.
    - _hba1c_%: Hemoglobin A1C percentage of the participant by the original dataset creators.
    - *CGM_type*: Continuous glucose monitoring (CGM) device used for data collection by the original dataset creators.
    - *glucose_level_record_count*: Total number of CGM recordings available for each participant in the dataset.
    - *average_glucose_level_mg_dl*: Mean glucose level (mg/dL) across all CGM recordings available for each participant in the dataset.
    - *count_days_with_CGM_data*: Total number of unique days on which at least one glucose reading was recorded for each participant in the dataset.

3. `D1NAMO-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`D1NAMO-extracted-glucose-files/`) and metadata (`D1NAMO_metadata.csv`) for the D1NAMO dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Dubosson, Fabien, Jean-Eudes Ranvier, Stefano Bromuri, Jean-Paul Calbimonte, Juan Ruiz, and Michael Schumacher. "The open D1NAMO dataset: A multi-modal dataset for research on non-invasive type 1 diabetes management." [https://zenodo.org/records/5651217](https://zenodo.org/records/5651217)

**License Details**
- The original dataset is distributed under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/legalcode) (CC BY-SA 4.0) license.

**Publication Reference(s)**
- Dubosson, Fabien, Jean-Eudes Ranvier, Stefano Bromuri, Jean-Paul Calbimonte, Juan Ruiz, and Michael Schumacher. “The Open D1NAMO Dataset: A Multi-Modal Dataset for Research on Non-Invasive Type 1 Diabetes Management.” Informatics in Medicine Unlocked 13 (January 2018): 92–100. [https://doi.org/10.1016/j.imu.2018.09.003](https://doi.org/10.1016/j.imu.2018.09.003)
