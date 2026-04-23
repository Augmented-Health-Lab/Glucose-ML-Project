# 6_Case-study

This directory contains the results of two case studies using open-access CGM datasets from the Glucose-ML collection. Also included are the Python scripts needed to replicate these analyses. Detailed instructions for running both case studies are included in this README.

The 2 Case studies:
- **Case_Study_1**: Trains three diabetes status classification models using 13 open-source datasets from the Glucose-ML collection.
- **Case_Study_2**: Comparative analysis of diabetes status classification performance when two sets of three models are trained using a single dataset vs multiple datasets from the Glucose-ML collection.


## `Case_Study_1/`
*Overview*: This first case study demonstrates a practical use case of harmonzied CGM datasets from the Glucose-ML collection to predict participant diabetes status (T1D, T2D, ND, PreD). Three common ML classification models (Logistic Regression, Random Forest, and XGBoost) are trained using 13 open-source datasets. Model performance was evaluated based on ability to classify diabetes status of participants from CGM-derived features.

**The following 13 Open-Access datasets were used:**
* AZT1D, BIGIDEAs, Bris-T1D_Open, CGMacros_Dexcom, Colas_2019, D1NAMO, Hall_2018, HUPA-UCM, PhysioCGM, ShanghaiT1DM, ShanghaiT2DM, T1D-UOM, UCHTT1DM

**Pre-Generated Model-Result Contents**:
* `Case_Study_1/Model-Results/`: Contains confusion matricies & performance scores for Logistic Regression, Random Forest, and XGBoost models trained on the 13 datasets.
* Additionally, the `Open-ML-Ready-Datasets/` (preprocessed data), `feature_calcs.csv` (features), and `participant_splits.csv` (train/split/validate assignments) are provided.

## `Case_Study_2/`
*Overview*: This second case study evaluates whether increasing the number of datasets (and therefore sample size and sample diversity) improves overall model performance in classifying diabetes status (T2D, ND, PreD) from CGM-derived features. Three common ML classification models (Logistic Regression, Random Forest, and XGBoost) are trained two times. The first triplet of models was trained using a single CGM dataset (CGMacros_Dexcom) and the second triplet was trained using five CGM datasets (CGMacros_Dexcom, Colas_2019, Hall_2018, ShanghaiT2DM, BIGIDEAs). Performance of the single-dataset and multi-dataset was compared.

**The following datasets were used for training**:
* **Single-dataset**: CGMacros_Dexcom
* **Multi-dataset**: CGMacros_Dexcom, Colas_2019, Hall_2018, ShanghaiT2DM, BIGIDEAs

**Pre-Generated Model-Result Contents**:
* `Single_Dataset_Model/Model-Results/`: Contains confusion matricies & performance scores for Logistic Regression, Random Forest, and XGBoost models trained on the CGMacros_Dexcom dataset.
* `Multi_Dataset_Model/Model-Results/`: Contains confusion matricies & performance scores for Logistic Regression, Random Forest, and XGBoost models trained on all five datasets combined.
* Additionally, the `Open-ML-Ready-Datasets/`, `feature_calcs.csv`, and `participant_splits.csv` inputs are provided in both the `Single_Dataset_Model/` and `Multi_Dataset_Model/` directories. 

## Script Structure

A total of 4 scripts need to be executed sequentially: 

1. `7_Open-ML-Resources/1_Split-participants.py`
2. `7_Open-ML-Resources/2_Preprocess-datasets.py` -> Note: `Open-ML-Ready-Datasets/` needs to be moved to `6_Case-study/` in order to proceed with step 3.
3. `Calculate-features.py`
4. `Run-case-study-models.py`


## Running the 4 Script Pipeline.

Note: The following steps assumes the user is starting from the `Glucose-ML-Project` directory & that harmonized CGM files and metadata exist in `3_Glucose-ML-collection/`.


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
python 1_Split-participants.py # see note below for important detail.
python 2_Preprocess-datasets.py
```
*IMPORTANT NOTE:* variable `open_datasets` at the top of the `1_Split-participants.py ` needs to be commented/uncommented to match the case study being executed.

4. Move the `Open-ML-Ready-Datasets` output created by `2_Preprocess-datasets.py` to `6_Case-study` using the following command:

```bash
mv Open-ML-Ready-Datasets ../6_Case-study/
```

5. Change your working directory to the following

```bash
cd ../6_Case-study
```

6. Finally, calcualte the features and train the models by executing the 2 scripts

```bash
python Calculate-features.py
python Run-case-study-models.py
```

---

## Overview of the Scripts

See [README](/7_Open-ML-Resources/README.md) for overview about `1_Split-participants.py` and `2_Preprocess-datasets.py`

### 1) Calculate-features.py

Calculates participant-level glucose features using up to 15 CGM days from the processed data.

**Input:**

* `Open-ML-Ready-Datasets/preprocessing_manifest.csv`
* `Open-ML-Ready-Datasets/[dataset]/[person_id].csv`

**Output:**

* `feature_calcs.csv`

CGM-derived features include summary statistics and glycemic variability measures such as:

* mean, median, SD glucose
* ADRR, MAGE, LBGI, HBGI, BGRI
* percent in various glucose ranges

---

### 2) Run-case-study-models.py

Trains and evaluates 3 models: logistic regression, random forest, and XGBoost using the calculated features.

**Input:**

* `feature_calcs.csv`

**Output:**

* `Model-Results/Logistic-regression/scores.csv`
* `Model-Results/Logistic-regression/confusion_matrix.csv`
* `Model-Results/Random-forest/scores.csv`
* `Model-Results/Random-forest/confusion_matrix.csv`
* `Model-Results/XGBoost/scores.csv`
* `Model-Results/XGBoost/confusion_matrix.csv`

Metrics included in "scores.csv": accuracy, macro F1, balanced accuracy


<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>