# PhysioCGM
The Glucose-ML Project curates and harmonizes CGM data and associated metadata from various datasets in the collection. This directory contains the harmonized glucose data and metadata for the *PhysioCGM* dataset.

## Contents

1. `PhysioCGM-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `PhysioCGM_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `PhysioCGM-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`PhysioCGM-extracted-glucose-files/`) and metadata (`PhysioCGM_metadata.csv`) for the PhysioCGM dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Quamer, Waris; Tseng, Mu-Ruei; Vyas, Kathan; Villegas, Carolina; McKay, Siripoom; DeSalvo, Daniel J.; et al. (2025). PhysioCGM: a multimodal physiological dataset for non-invasive blood glucose estimation. figshare. Dataset. [https://doi.org/10.6084/m9.figshare.28136294.v1](https://doi.org/10.6084/m9.figshare.28136294.v1)

**License Details**
- The original dataset is licensed under a [Creative Commons CC0 1.0 Universal License](https://creativecommons.org/publicdomain/zero/1.0/)

**Publication Reference(s)**
- Quamer, W., Tseng, MR., Vyas, K. et al. A multimodal physiological dataset for non-invasive blood glucose estimation. Sci Data 12, 1822 (2025). [https://doi.org/10.1038/s41597-025-06090-6](https://doi.org/10.1038/s41597-025-06090-6)