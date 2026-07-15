# HUPA-UCM
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *HUPA-UCM* dataset.

## Contents

1. `HUPA-UCM-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `HUPA-UCM_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `HUPA-UCM-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`HUPA-UCM-extracted-glucose-files/`) and metadata (`HUPA-UCM_metadata.csv`) for the HUPA-UCM dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Hidalgo, J. Ignacio; Alvarado, Jorge; Botella, Marta; Aramendi, Aranzazu; Velasco, J. Manuel; Garnica, Oscar (2024), “HUPA-UCM Diabetes Dataset”, Mendeley Data, V1, doi: 10.17632/3hbcscwz44.1

**License Details**
- The original dataset is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)

**Publication Reference(s)**
- Hidalgo, J. Ignacio, Jorge Alvarado, Marta Botella, Aranzazu Aramendi, J. Manuel Velasco, and Oscar Garnica. “HUPA-UCM Diabetes Dataset.” Data in Brief 55 (August 2024): 110559. [https://doi.org/10.1016/j.dib.2024.110559](https://doi.org/10.1016/j.dib.2024.110559)