# ShanghaiT1DM
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *ShanghaiT1DM* dataset.

## Contents

1. `ShanghaiT1DM-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `ShanghaiT1DM_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `ShanghaiT1DM-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`ShanghaiT1DM-extracted-glucose-files/`) and metadata (`ShanghaiT1DM_metadata.csv`) for the ShanghaiT1DM dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Zhao, Qinpei; Zhu, Jinhao; Wang, Congrong; Rao, Weixiong (2022). Diabetes Datasets, ShanghaiT1DM and ShanghaiT2DM. figshare. Collection. [https://doi.org/10.6084/m9.figshare.c.6310860](https://doi.org/10.6084/m9.figshare.c.6310860)

**License Details**
- No dataset license was found. The associated publication is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)

**Publication Reference(s)**
- Zhao, Q., Zhu, J., Shen, X. et al. Chinese diabetes datasets for data-driven machine learning. Sci Data 10, 35 (2023). [https://doi.org/10.1038/s41597-023-01940-7](https://doi.org/10.1038/s41597-023-01940-7)
