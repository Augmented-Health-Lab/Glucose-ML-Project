# AZT1D
This directory contains the harmonized glucose data and metadata for the *AZT1D* project.

## Downloading the Harmonized Data

To download the harmonized AZT1D files WITHOUT the [download script](/1_Auto-scripts/auto-download-open-datasets.py), please download and extract the *AZT1D-from-Glucose-ML.zip* file.

## Download Contents

1. `{Dataset}/AZT1D-extracted-glucose-files/`:
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

## Data License & Citation Information

**Where to find the original dataset?**
- Saman Khamesian, Asiful Arefeen, Bithika M. Thompson, Adela Grando, and Hassan Ghasemzadeh. 2025. AZT1D: A Real-World Dataset for Type 1 Diabetes. [https://doi.org/10.17632/gk9m674wcx.1](https://doi.org/10.17632/gk9m674wcx.1)

**License Details**
- The original dataset is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)

**Publication Reference(s)**
- S. Khamesian, A. Arefeen, B. M. Thompson, M. A. Grando and H. Ghasemzadeh, "AZT1D: A Real-World Dataset for Type 1 Diabetes," 2025 IEEE 21st International Conference on Body Sensor Networks (BSN), Los Angeles, CA, USA, 2025, pp. 1-4. [https://ieeexplore.ieee.org/document/11337496](https://ieeexplore.ieee.org/document/11337496)

