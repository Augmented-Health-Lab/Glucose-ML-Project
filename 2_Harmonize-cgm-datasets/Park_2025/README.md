# Park_2025 - Helper Scripts

The Glucose-ML Project develops and provide helper scripts to extract and harmonize CGM data and associated metadata from various datasets in the collection. This directory contains the openly available helper scripts that can be used for the *Park_2025* dataset. 

---

## Contents

* `Park_2025_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized CGM CSV files, providing one .csv file per participant. See expected output files [here](../../3_Glucose-ML-collection/Park_2025/Park_2025-extracted-glucose-files).
* `Park_2025_metadata.py` - Calculates participant-level summary statistics from the standardized CGM files. See expected output file [here](../../3_Glucose-ML-collection/Park_2025/Park_2025-metadata.csv).
    - *NOTE*: The metadata.py script calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata including diabetes_type, age, gender, race_ethnicity, hba1c_%, and CGM_type was manually curated from the original data source. 

Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Download the Park_2025 dataset from the original data source (https://cgmdb.stanford.edu/data/) or use our auto_download scripts [here](../../1_Auto-scripts). The file you need to download is called *data_cgm.csv*. If the download is a zipped file, DO NOT UNZIP.
2. Place the unzipped download in `1_Auto-scripts/Original-Glucose-ML-datasets/Park_2025_raw_data/` (Make the directory, if needed.)
3. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/Park_2025
python Park_2025_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/Park_2025_raw_data/
```
4. Harmonized csv files are written to: `Standardized-datasets/Park_2025/<subject_id>.csv`
5. To calculate the participant-level metadata, run the following:

```bash
python Park_2025_metadata.py Standardized-datasets/Park_2025
```

