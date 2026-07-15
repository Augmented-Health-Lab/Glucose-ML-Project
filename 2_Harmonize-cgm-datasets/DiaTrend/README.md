# DiaTrend - Helper Scripts

The Glucose-ML Project develops and provide helper scripts to extract and harmonize CGM data and associated metadata from various datasets in the collection. This directory contains the openly available helper scripts that can be used for the *DiaTrend* dataset. 

*NOTE*: DiaTrend is a controlled-access dataset. Existing data use agreements (DUA) do not permit access to the extracted glucose files and metadata for this dataset. Please request access at [https://doi.org/10.7303/syn38187184](https://doi.org/10.7303/syn38187184), after which you can proceed with using our helper scripts.

---

## Contents

* `DiaTrend_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized CGM CSV files, providing one .csv file per participant. 
* `DiaTrend_metadata.py` - Calculates participant-level summary statistics from the standardized CGM files. 
    - *NOTE*: The metadata.py script calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata including diabetes_type, age, gender, race_ethnicity, hba1c_%, and CGM_type was manually curated from the original data source. 

Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Request access and download the DiaTrend dataset from the original data source (https://doi.org/10.7303/syn38187184). If the download is a zipped file, DO NOT UNZIP.
2. Place the unzipped download in `1_Auto-scripts/Original-Glucose-ML-datasets/DiaTrend_raw_data/` (Make the directory, if needed.)
3. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/DiaTrend
python DiaTrend_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/DiaTrend_raw_data/
```
4. Harmonized csv files are written to: `Standardized-datasets/DiaTrend/<subject_id>.csv`
5. To calculate the participant-level metadata, run the following:

```bash
python DiaTrend_metadata.py Standardized-datasets/DiaTrend
```

