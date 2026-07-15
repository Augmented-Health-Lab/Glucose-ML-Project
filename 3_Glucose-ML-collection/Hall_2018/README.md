# Hall_2018
The [Glucose-ML Project](https://www.glucose-ml-project.com/) curates and harmonizes public CGM datasets and associated metadata for research. This directory contains the harmonized glucose data and metadata for the *Hall_2018* dataset.

## Contents

1. `Hall_2018-extracted-glucose-files/`: This directory contains .cvs files for each participant in the dataset. Each individual participant file includes:
    - _timestamp_: The time in which the glucose reading was collected.
    - *glucose_value_mg_dl*: The CGM glucose reading in mg/dL.

2. `Hall_2018_metadata.csv`: This file contains a summary of metadata from all participants in this dataset. The columns included are as follows:
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

3. `Hall_2018-from-Glucose-ML.zip`: This zip file contains the harmonized glucose data (`Hall_2018-extracted-glucose-files/`), metadata (`Hall_2018_metadata.csv`), and this README for the Hall_2018 dataset and is included so the user can download the data directly from our GitHub repository.

**Where to find the original dataset?**
- Dataset is available within the paper and its supplemenaty materials.

**License Details**
- The original dataset is distributed under the publication, which is licensed under the [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) license.

**Publication Reference(s)**
- Hall, H., Perelman, D., Breschi, A., Limcaoco, P., Kellogg, R., McLaughlin, T., & Snyder, M. (2018). Glucotypes reveal new patterns of glucose dysregulation. PLoS biology, 16(7), e2005143. [https://doi.org/10.1371/journal.pbio.2005143](https://doi.org/10.1371/journal.pbio.2005143)