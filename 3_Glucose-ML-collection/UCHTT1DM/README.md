# UCHTT1DM
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *UCHTT1DM* dataset.

## Contents

1. `UCHTT1DM-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `UCHTT1DM_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `UCHTT1DM-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`UCHTT1DM-extracted-glucose-files/`) and metadata (`UCHTT1DM_metadata.csv`) for the UCHTT1DM dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Saúl Langarica, Diego de la Vega, Nawel Cariman, Martín Miranda, David C. Andrade, Felipe Núñez, Maria Rodriguez-Fernandez. Deep Learning-Based Glucose Prediction Models: A Guide for Practitioners and a Curated Dataset for Improved Diabetes Management. IEEE Open Journal of Engineering in Medicine and Biology, 2023 (Under review). [https://github.com/fisiologiacuantitativauc/UC_HT_T1DM](https://github.com/fisiologiacuantitativauc/UC_HT_T1DM)

**License Details**
- No dataset license was found. The associated publication is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/) (CC BY-NC-ND 4.0)

**Publication Reference(s)**
- S. Langarica et al., "Deep Learning-Based Glucose Prediction Models: A Guide for Practitioners and a Curated Dataset for Improved Diabetes Management," in IEEE Open Journal of Engineering in Medicine and Biology, vol. 5, pp. 467-475, 2024, doi: 10.1109/OJEMB.2024.3365290.