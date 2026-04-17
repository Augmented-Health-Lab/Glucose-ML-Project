# 7_ML-Tools

This directory contains ML-preprocessing tools used to prepare Continuous Glucose Monitoring (CGM) data for machine learning workflows in the Glucose-ML project.

These tools handle participant-level data splitting and preprocessing to generate standardized, model-ready datasets.

---

## Overview

At present, there are two ML-tool scripts available for use

1. `1_Split-participants.py`
2. `2_Preprocess-datasets.py`

---

## Script Summary

### 1) 1_Split-participants.py

Splits participants into training, validation, and test sets. Splitting operates within each dataset independently and uses a fixed random seed for reproducibility.

**Input:**
* `3_Glucose-ML-collection/[dataset]/[dataset]-metadata.csv`

**Output:**
* `participant_splits.csv`

For each dataset:
* Performs stratified splitting by diabetes type.
* Splits datasets specified by variable **open_projects**. Note: User can add as many datasets they have data for so long as datasets > 1.
* Applies a 70/10/20 split (train/validate/test). Note: User can modify these values.


---

### 2) 2_Preprocess-datasets.py

Preprocesses harmonized CGM data (from 3_Glucose-ML-collection) into a standardized format for machine learning.

**Input:**
* `participant_splits.csv`
* `3_Glucose-ML-collection/[dataset]/[dataset]-extracted-glucose-files/*.csv`

**Output:**
* `ML-Ready-Datasets/[dataset]/[person_id].csv`
* `ML-Ready-Datasets/preprocessing_manifest.csv`

For each participant:
* Loads standardized glucose data according to `participant_splits.csv`
* Resamples to 5-minute intervals
* Interpolates small gaps (up to 15 minutes)
* Filters out low-quality CGM days (<70% coverage)

Note: Participants who do not have any valid CGM data for ML analysis post-processing will have 'no' under the 'passed' column in the `preprocessing_manifest.csv`

---

## Running these Scripts

Run the scripts in order:

```bash
python 1_Split-participants.py
python 2_Preprocess-datasets.py
```


## Output

After running both scripts, the `ML-Ready-Datasets/` directory will contain:

* Cleaned CGM files for each participant
* A preprocessing manifest summarizing data quality and processing status. 

---

We hope these open-resources enable quick reproduction and experimentaion for downstream AI/ML tasks.

<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>
