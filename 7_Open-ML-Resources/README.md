# 7_Open-ML-Resources

This directory contains tools for preprocessing CGM data from the Glucose-ML collection into standardized, machine learning–ready formats.

These scripts perform participant-level splitting and data preprocessing to generate standardized, model-ready datasets for downstream AI/ML tasks.

---

## Script Summary

### 1) 1_Split-participants.py

Splits participants into training, validation, and test sets. Splitting operates within each dataset independently and uses a fixed random seed (20) for reproducibility.

**Input:**
* `3_Glucose-ML-collection/[dataset]/[dataset]-metadata.csv`

**Output:**
* `participant_splits.csv`

For each dataset:
* Performs stratified splitting by diabetes type.
* Splits datasets specified by variable **open_projects**.
* Applies a 70/10/20 split (train/validate/test). Note: User can modify these values.


### 2) 2_Preprocess-datasets.py

Preprocesses harmonized CGM data (from 3_Glucose-ML-collection) into a standardized format for machine learning.

**Input:**
* `participant_splits.csv`
* `3_Glucose-ML-collection/[dataset]/[dataset]-extracted-glucose-files/*.csv`

**Output:**
* `Open-ML-Ready-Datasets/[dataset]/[person_id].csv`
* `Open-ML-Ready-Datasets/preprocessing_manifest.csv`

For each participant:
* Loads standardized glucose data according to `participant_splits.csv`
* Resamples to 5-minute intervals
* Interpolates small gaps (up to 15 minutes)
* Filters out low-quality CGM days (<70% coverage)

Note: Participants who do not have any valid CGM data for ML analysis post-processing will have 'no' under the 'passed' column in the `preprocessing_manifest.csv`

---

## Running these Scripts
Note: The following steps assumes the user is starting from the Glucose-ML-Project directory & harmonized CGM files and metadata are avaible in `3_Glucose-ML-collection/`.

2. Change your working directory to the following

```bash
cd 7_Open-ML-Resources
```

2. Run the scripts in order:

```bash
python 1_Split-participants.py
python 2_Preprocess-datasets.py
```
Note: These two scripts are used in both case studies as listed here [README](/6_Case-study/README.md)

Outputs: After running both scripts, the `Open-ML-Ready-Datasets/` directory will contain:

* Cleaned CGM files for each participant
* A preprocessing manifest summarizing data quality and processing status. 

---

We hope these open-resources enable quick reproduction and experimentation for downstream AI/ML tasks.

<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>
