# UCHTT1DM - Helper Scripts

The Glucose-ML Project develops and provide helper scripts to extract and harmonize CGM data and associated metadata from various datasets in the collection. This directory contains the openly available helper scripts that can be used for the *UCHTT1DM* dataset. 

---

## Contents

* `UCHTT1DM_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized CGM CSV files, providing one .csv file per participant. See expected output files [here](../../3_Glucose-ML-collection/UCHTT1DM/UCHTT1DM-extracted-glucose-files).
* `UCHTT1DM_metadata.py` - Calculates participant-level summary statistics from the standardized CGM files. See expected output file [here](../../3_Glucose-ML-collection/UCHTT1DM/UCHTT1DM-metadata.csv).
    - *NOTE*: The metadata.py script calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata including diabetes_type, age, gender, race_ethnicity, hba1c_%, and CGM_type was manually curated from the original data source. 

Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Download the UCHTT1DM dataset from the original data source (https://github.com/fisiologiacuantitativauc/UC_HT_T1DM) or use our auto_download scripts [here](../../1_Auto-scripts).
2. Place the download contents in `1_Auto-scripts/Original-Glucose-ML-datasets/UCHTT1DM_raw_data/` (Make the directory, if needed.)
3. If the downloaded file is a ZIP archive, extract it. If the archive contains additional nested ZIP files, extract those as well.
4. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/UCHTT1DM
python UCHTT1DM_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/UCHTT1DM_raw_data/
```
5. Harmonized csv files are written to: `Standardized-datasets/UCHTT1DM/<subject_id>.csv`
6. To calculate the participant-level metadata, run the following:

```bash
python UCHTT1DM_metadata.py Standardized-datasets/UCHTT1DM
```

