# 6_Case-study

This directory contains a four-step workflow for building diabetes classification models using the Open-Access Glucose-ML standardized continuous glucose monitoring (CGM) data.

See `1_Dataset_VS_5_Datasets_Comparison/` for the results of our Single vs Multi-dataset Analysis. For more info see [README](1_Dataset_VS_5_Datasets_Comparison/README.md)

## Script Structure

A total of 4 scripts need to be run in the following order: 

1. `7_Open-ML-Resources/1_Split-participants.py`
2. `7_Open-ML-Resources/2_Preprocess-datasets.py` -> Note: `Open-ML-Ready-Datasets/` needs to be moved to `6_Case-study` to proceed with step 3.
3. `Calculate-features.py`
4. `Run-case-study-models.py`

---

## Inputs

This pipeline assumes the base Glucose-ML file structure and accesses the CGM files and metadata from 3_Glucose-ML-collection. The only exception is `Open-ML-Ready-Datasets/` as stated in **Script Structure**.

---

## Reproducing this Case-Study

The following steps is assumed the user is in the Glucose-ML-Project directory.

1. Install the following dependencies if you don't already have them

```bash
pip install pandas numpy scikit-learn xgboost
```

2. Change your working directory to the following

```bash
cd 7_Open-ML-Resources
```
3. Run these 2 scripts! 

```bash
python 1_Split-participants.py
python 2_Preprocess-datasets.py
```
4. Move the output from `2_Preprocess-datasets.py` to `6_Case-study` using the following command

```bash
mv Open-ML-Ready-Datasets ../6_Case-study/
```

5. Change your working directory to the following

```bash
cd 6_Case-study
```

6. Finally, run the final 2 scripts

```bash
python Calculate_features.py
python Run-case-study-models.py
```

---

## Script Summary

See [README](/7_Open-ML-Resources/README.md) for information about `1_Split-participants.py` and `2_Preprocess-datasets.py`

### 1) Calculate-features.py

Calculates participant-level glucose features from the processed data.

* Specify up to as many valid CGM days per participant (default is 15).

**Input:**

* `Open-ML-Ready-Datasets/preprocessing_manifest.csv`
* `Open-ML-Ready-Datasets/[dataset]/[person_id].csv`

**Output:**

* `feature_calcs.csv`

Features include summary statistics and glycemic variability measures such as:

* mean, median, SD glucose
* ADRR, MAGE, LBGI, HBGI, BGRI
* percent in various glucose ranges

---

### 2) Run-case-study-models.py

Trains and evaluates 3 models: logistic regression, random forest, and XGBoost using the calculated features.

**Input:**

* `feature_calcs.csv`

**Output:**

* `Logistic-regression-results/test_scores.csv`
* `Logistic-regression-results/test_confusion_matrix.csv`
* `Random-forest-results/test_scores.csv`
* `Random-forest-results/test_confusion_matrix.csv`
* `XGBoost-results/test_scores.csv`
* `XGBoost-results/test_confusion_matrix.csv`

Metrics included in "test_scores.csv": accuracy, macro F1, balanced accuracy


<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>