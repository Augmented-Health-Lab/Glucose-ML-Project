# Colas_2019
The [Glucose-ML Project](https://www.glucose-ml-project.com/) curates and harmonizes public CGM datasets and associated metadata for research. This directory contains the harmonized glucose data and metadata for the *Colas_2019* dataset.

## Contents

1. `Colas_2019-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `Colas_2019_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `Colas_2019-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`Colas_2019-extracted-glucose-files/`), metadata (`Colas_2019_metadata.csv`), and this README for the Colas_2019 dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Dataset is available within the paper and its supplemenaty materials.

**License Details**
- The original dataset is distributed under the publication, which is licensed under the [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) license.

**Publication Reference(s)**
- Colás, A., Vigil, L., Vargas, B., Cuesta-Frau, D., & Varela, M. (2019). Detrended Fluctuation Analysis in the prediction of type 2 diabetes mellitus in patients at risk: Model optimization and comparison with other metrics. PloS one, 14(12), e0225817. [https://doi.org/10.1371/journal.pone.0225817](https://doi.org/10.1371/journal.pone.0225817)

