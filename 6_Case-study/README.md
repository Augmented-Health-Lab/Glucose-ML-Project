# 6_Case-study

This directory contains a four-step workflow for building a diabetes classification model from the Open-Access Glucose-ML standardized continuous glucose monitoring (CGM) data.

## Workflow

There are a total of 4 scripts that should be run in the following order: 

1. `1_split_participants.py`
2. `2_preprocess_data.py`
3. `3_calculate_features.py`
4. `4_machine_learning.py`

---

## Inputs

This pipeline assumes the base Glucose-ML file structure and accesses the CGM files and metadata from 3_Glucose-ML-collection.

---

## Getting Started

1. Install the following dependencies if you don't already have them

```bash
pip install pandas numpy scikit-learn
```

2. Change your working directory to the following

```bash
cd 6_Case-study
```
3. Run the scripts! 

```bash
python 1_split_participants.py
python 2_preprocess_data.py
python 3_calculate_features.py
python 4_machine_learning.py
```

---

## Script Summary

### `1_split_participants.py`

Splits participants within each dataset by `diabetes_type` into:

* 70% train
* 10% validation
* 20% test

**Input:**
`3_Glucose-ML-collection/[dataset]/[dataset]-metadata.csv`

**Output:**
`participant_splits.csv`

---

### `2_preprocess_data.py`

Preprocesses glucose files by:

* resampling to 5-minute intervals
* interpolating small gaps up to 15 minutes
* keeping days with at least 70% coverage
* keeping up to as many valid days per participant (default is 15).

**Input:**

* `participant_splits.csv`
* `3_Glucose-ML-collection/[dataset]/[dataset]-extracted-glucose-files/*.csv`

**Output:**

* `Processed-Data/[dataset]/[person_id].csv`
* `Processed-Data/preprocessing_manifest.csv`

---

### `3_calculate_features.py`

Calculates participant-level glucose features from the processed data.

**Input:**

* `Processed-Data/preprocessing_manifest.csv`
* `Processed-Data/[dataset]/[person_id].csv`

**Output:**

* `feature_calcs.csv`

Features include summary statistics and glycemic variability measures such as:

* mean glucose
* median glucose
* SD
* CV
* MAGE
* ADRR
* LBGI / HBGI / BGRI
* percent in glucose ranges

---

### `4_machine_learning.py`

Trains and evaluates a logistic regression model using the calculated features.

**Input:**
`feature_calcs.csv`

**Output:**

* `Logistic-regression-results/test_scores.csv`
* `Logistic-regression-results/test_confusion_matrix.csv`

Metrics include:

* accuracy
* macro F1
* balanced accuracy




<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>