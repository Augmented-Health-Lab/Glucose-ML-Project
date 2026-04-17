# 6_Case-study: Single-Dataset VS. Multi-Dataset Model Comparison

## Overview

This experiment sets out to determine whether increasing the number of datasets (and therefore sample size and diversity) improves model performance in classifying diabetes status from CGM-derived features.

We compare:

* **Single-dataset model**: CGMacros_Dexcom
* **Multi-dataset model**: CGMacros_Dexcom, Colas_2019, Hall_2018, ShanghaiT2DM, BIGIDEAs

All models were trained to classify:

* No Diabetes (ND)
* Prediabetes (PreD)
* Type 2 Diabetes (T2D)

**Important Note**: This experiment is intended as a baseline comparison rather than a strict generalization benchmark

---

## Directory Structure & Results

```
1_Dataset_VS_5_Datasets_Comparison/
├── Combined_Dataset_Models/
├── CGMacros_Models/
└── README.md
```

* `CGMacros_Models/`: Contains results from Logistic Regression, Random Forest, and XGBoost models trained using only the CGMacros_Dexcom dataset.
* `Combined_Dataset_Models/`: Contains results from Logistic Regression, Random Forest, and XGBoost models trained using all five datasets combined.

Results for each of the Models:
  - `scores.csv`: accuracy, macro F1, and balanced accuracy
  - `confusion_matrix.csv`: confusion matrix for test predictions

---

## Experimental Setup

Protocol listed in 

* Features derived from standardized CGM data and 15 CGM days max.
* Consistent preprocessing and feature engineering across all datasets (seed = 20)
* Train (70%) / Validation (10%) / Test (20%) split applied within each dataset
* Same model configurations used for both experiments

---

<p>&nbsp;</p>

<p align="center">
  <img src="../../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>