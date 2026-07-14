# AZT1D - Harmonization Scripts

This directory contains the helper scripts needed to harmonize the *AZT1D* projects CGM reads and metadata. 

*Note*: The harmonization process is made much easier for open-access datasets via our [auto-harmonizer pipeline](/1_Auto-scripts/README.md), but can still be executed individually if desired.

---

## Contents

* `AZT1D_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized, per-subject CGM CSV files.
* `AZT1D_metadata.py` - Calculates subject-level summary statistics from the standardized CGM files.

*Note*: Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Download the raw AZT1D dataset (which can be found [here](/3_Glucose-ML-collection/AZT1D/README.md)). If the download is a zipped file, DO NOT UNZIP.
2. Place the unzipped download in `1_Auto-scripts/Original-Glucose-ML-datasets/AZT1D_raw_data/` (Make the directory, if needed.)
3. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/AZT1D
python AZT1D_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/AZT1D_raw_data/
```
4. Harmonized csv files are written to: `Standardized-datasets/AZT1D/<subject_id>.csv`
5. To calculate the individual-level metadata, run the following:

```bash
python AZT1D_metadata.py Standardized-datasets/AZT1D
```

*NOTE*: The metadata.py script only calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata for the AZT1D project can be found [here](/3_Glucose-ML-collection/AZT1D/AZT1D-metadata.csv)