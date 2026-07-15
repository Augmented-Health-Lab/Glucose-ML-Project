# Bris-T1D_Open
The [Glucose-ML Project](https://www.glucose-ml-project.com/) curates and harmonizes public CGM datasets and associated metadata for research. This directory contains the harmonized glucose data and metadata for the *Bris-T1D_Open* dataset.

## Contents

1. `Bris-T1D_Open-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `Bris-T1D_Open_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `Bris-T1D_Open-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`Bris-T1D_Open-extracted-glucose-files/`), metadata (`Bris-T1D_Open_metadata.csv`), and this README for the Bris-T1D_Open dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Sam Gordon James, Miranda Armstrong, Aisling O'Kane, Harry Emerson, Zahraa Abdallah (2025): BrisT1D-Open Dataset. [https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08](https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08)

**License Details**
- The original dataset is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)

**Publication Reference(s)**
- Sam Gordon James, Miranda Elaine Glynis Armstrong, Aisling Ann O’Kane, Harry Emerson, and Zahraa S Abdallah. 2025. BrisT1D Dataset: Young Adults with Type 1 Diabetes in the UK using Smartwatches. [https://arxiv.org/abs/2507.17757](https://arxiv.org/abs/2507.17757)

