# Hall_2018 - Helper Scripts

The Glucose-ML Project develops and provide helper scripts to extract and harmonize CGM data and associated metadata from various datasets in the collection. This directory contains the openly available helper scripts that can be used for the *Hall_2018* dataset. 

---

## Contents

* `Hall_2018_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized CGM CSV files, providing one .csv file per participant. See expected output files [here](../../3_Glucose-ML-collection/Hall_2018/Hall_2018-extracted-glucose-files).
* `Hall_2018_metadata.py` - Calculates participant-level summary statistics from the standardized CGM files. See expected output file [here](../../3_Glucose-ML-collection/Hall_2018/Hall_2018-metadata.csv).
    - *NOTE*: The metadata.py script calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata including diabetes_type, age, gender, race_ethnicity, hba1c_%, and CGM_type was manually curated from the original data source. 

Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Download the Hall_2018 dataset from the original data source (https://pubmed.ncbi.nlm.nih.gov/30040822/) or use our auto_download scripts [here](../../1_Auto-scripts). If the download is a zipped file, DO NOT UNZIP.
2. Place the unzipped download in `1_Auto-scripts/Original-Glucose-ML-datasets/Hall_2018_raw_data/` (Make the directory, if needed.)
3. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/Hall_2018
python Hall_2018_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/Hall_2018_raw_data/
```
4. Harmonized csv files are written to: `Standardized-datasets/Hall_2018/<subject_id>.csv`
5. To calculate the participant-level metadata, run the following:

```bash
python Hall_2018_metadata.py Standardized-datasets/Hall_2018
```

