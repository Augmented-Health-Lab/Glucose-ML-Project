# CGMacros_Libre
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *CGMacros_Libre* dataset.

## Contents

1. `CGMacros_Libre-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `CGMacros_Libre_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `CGMacros_Libre-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`CGMacros_Libre-extracted-glucose-files/`) and metadata (`CGMacros_Libre_metadata.csv`) for the CGMacros_Libre dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. RRID:SCR_007345. [https://doi.org/10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658)

**License Details**
- The original dataset is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0)

**Publication Reference(s)**
- Das, A., Kerr, D., Glantz, N. et al. CGMacros: a pilot scientific dataset for personalized nutrition and diet monitoring. Sci Data 12, 1557 (2025). [https://doi.org/10.1038/s41597-025-05851-7](https://doi.org/10.1038/s41597-025-05851-7)

