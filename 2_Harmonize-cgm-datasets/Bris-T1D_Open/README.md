# Bris-T1D_Open - Helper Scripts

The Glucose-ML Project develops and provide helper scripts to extract and harmonize CGM data and associated metadata from various datasets in the collection. This directory contains the openly available helper scripts that can be used for the *Bris-T1D_Open* dataset. 

---

## Contents

* `Bris-T1D_Open_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized CGM CSV files, providing one .csv file per participant. See expected output files [here](../../3_Glucose-ML-collection/Bris-T1D_Open/Bris-T1D_Open-extracted-glucose-files).
* `Bris-T1D_Open_metadata.py` - Calculates participant-level summary statistics from the standardized CGM files. See expected output file [here](../../3_Glucose-ML-collection/Bris-T1D_Open/Bris-T1D_Open-metadata.csv).
    - *NOTE*: The metadata.py script calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata including diabetes_type, age, gender, race_ethnicity, hba1c_%, and CGM_type was manually curated from the original data source. 

Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Download the Bris-T1D_Open dataset from the original data source (https://data.bris.ac.uk/data/dataset/33z5jc8fa6tob21ptrugzqog08) or use our auto_download scripts [here](../../1_Auto-scripts).
2. Place the download contents in `1_Auto-scripts/Original-Glucose-ML-datasets/Bris-T1D_Open_raw_data/` (Make the directory, if needed.)
3. If the downloaded file is a ZIP archive, extract it. If the archive contains additional nested ZIP files, extract those as well.
4. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/Bris-T1D_Open
python Bris-T1D_Open_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/Bris-T1D_Open_raw_data/
```
5. Harmonized csv files are written to: `Standardized-datasets/Bris-T1D_Open/<subject_id>.csv`
6. To calculate the participant-level metadata, run the following:

```bash
python Bris-T1D_Open_metadata.py Standardized-datasets/Bris-T1D_Open
```
