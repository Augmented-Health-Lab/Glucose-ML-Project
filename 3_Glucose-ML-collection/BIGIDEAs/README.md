# BIGIDEAs
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *BIGIDEAs* dataset.

## Contents

1. `BIGIDEAs-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `BIGIDEAs_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `BIGIDEAs-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`BIGIDEAs-extracted-glucose-files/`) and metadata (`BIGIDEAs_metadata.csv`) for the BIGIDEAs dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Cho, P., Kim, J., Bent, B., & Dunn, J. (2023). BIG IDEAs Lab Glycemic Variability and Wearable Device Data (version 1.1.2). PhysioNet. RRID:SCR_007345. [https://doi.org/10.13026/zthx-5212](https://doi.org/10.13026/zthx-5212)

**License Details**
- The original dataset is made available under the [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1.0/) (ODC-By) v1.0 

**Publication Reference(s)**
- Bent, B., Cho, P.J., Henriquez, M. et al. Engineering digital biomarkers of interstitial glucose from noninvasive smartwatches. npj Digit. Med. 4, 89 (2021). [https://doi.org/10.1038/s41746-021-00465-w](https://doi.org/10.1038/s41746-021-00465-w)

