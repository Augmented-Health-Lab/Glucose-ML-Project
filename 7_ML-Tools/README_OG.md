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

### `1_Split-participants.py`

Splits participants into training, validation, and test sets.

* Performs **stratified splitting by diabetes type**
* Applies a **70 / 10 / 20 split** (train / validate / test)
* Uses a fixed random seed for reproducibility
* Operates **within each dataset independently**

**Input:**

* `3_Glucose-ML-collection/[dataset]/[dataset]-metadata.csv`

**Output:**

* `participant_splits.csv`

---

### 2_Preprocess-datasets.py

Preprocesses raw CGM data into a standardized format for machine learning.

For each participant:

* Loads raw glucose data
* Resamples to **5-minute intervals**
* Interpolates small gaps (up to **15 minutes**)
* Filters out low-quality days (< **70% coverage**)
* Saves cleaned data to disk

Also generates a manifest summarizing preprocessing results.

**Inputs:**

* `participant_splits.csv`
* `3_Glucose-ML-collection/[dataset]/[dataset]-extracted-glucose-files/*.csv`

**Outputs:**

* `ML-Ready-Datasets/[dataset]/*.csv`
* `ML-Ready-Datasets/preprocessing_manifest.csv`

---

## How to Run

Run the scripts in order:

```bash
python 1_Split-participants.py
python 2_Preprocess-datasets.py
```

---

## Notes

* All participant IDs are handled as strings to preserve formatting (e.g., leading zeros).
* Preprocessing assumes a target sampling rate of **5 minutes (288 readings/day)**.
* Interpolation is limited to short gaps to avoid introducing artificial data.
* Only high-quality CGM days are retained for downstream analysis.

---

## Output

After running both scripts, the `ML-Ready-Datasets/` directory will contain:

* Cleaned CGM files for each participant
* A preprocessing manifest summarizing data quality and processing status

---

This directory serves as the foundation for downstream feature extraction and machine learning modeling.

<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>