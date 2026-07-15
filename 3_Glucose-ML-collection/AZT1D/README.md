# AZT1D
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *AZT1D* dataset.

## Contents

1. `{Dataset}/AZT1D-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `{Dataset}/{Dataset}_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

## Data License & Citation Information

**Where to find the original dataset?**
- Saman Khamesian, Asiful Arefeen, Bithika M. Thompson, Adela Grando, and Hassan Ghasemzadeh. 2025. AZT1D: A Real-World Dataset for Type 1 Diabetes. [https://doi.org/10.17632/gk9m674wcx.1](https://doi.org/10.17632/gk9m674wcx.1)

**License Details**
- The original dataset is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)

**Publication Reference(s)**
- S. Khamesian, A. Arefeen, B. M. Thompson, M. A. Grando and H. Ghasemzadeh, "AZT1D: A Real-World Dataset for Type 1 Diabetes," 2025 IEEE 21st International Conference on Body Sensor Networks (BSN), Los Angeles, CA, USA, 2025, pp. 1-4. [https://ieeexplore.ieee.org/document/11337496](https://ieeexplore.ieee.org/document/11337496)

