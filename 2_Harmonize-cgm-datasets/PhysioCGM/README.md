# PhysioCGM - Helper Scripts

The Glucose-ML Project develops and provide helper scripts to extract and harmonize CGM data and associated metadata from various datasets in the collection. This directory contains the openly available helper scripts that can be used for the *PhysioCGM* dataset. 

---

## Contents

* `PhysioCGM_extract-glucose-data.py` - Cleans and converts raw dataset files into standardized CGM CSV files, providing one .csv file per participant. See expected output files [here](../../3_Glucose-ML-collection/PhysioCGM/PhysioCGM-extracted-glucose-files).
* `PhysioCGM_metadata.py` - Calculates participant-level summary statistics from the standardized CGM files. See expected output file [here](../../3_Glucose-ML-collection/PhysioCGM/PhysioCGM-metadata.csv).
    - *NOTE*: The metadata.py script calculates the following data: glucose_level_record_count, average_glucose_level_mg_dl, count_days_with_CGM_data. Additional metadata including diabetes_type, age, gender, race_ethnicity, hba1c_%, and CGM_type was manually curated from the original data source. 

Additional information about these scripts can be found in this [README](/2_Harmonize-cgm-datasets/README.md)

---

## Executing the Harmonization Scripts Individually

Steps to execute the scripts:

1. Download the PhysioCGM dataset from the original [https://springernature.figshare.com/articles/dataset/PhysioCGM_a_multimodal_physiological_dataset_for_non-invasive_blood_glucose_estimation/28136294](https://springernature.figshare.com/articles/dataset/PhysioCGM_a_multimodal_physiological_dataset_for_non-invasive_blood_glucose_estimation/28136294) or use our auto_download scripts [here](../../1_Auto-scripts).
2. Place the download contents in `1_Auto-scripts/Original-Glucose-ML-datasets/PhysioCGM_raw_data/` (Make the directory, if needed.)
3. If the downloaded file is a ZIP archive, extract it. If the archive contains additional nested ZIP files, extract those as well.
4. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/PhysioCGM
python PhysioCGM_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/PhysioCGM_raw_data/
```
5. Harmonized csv files are written to: `Standardized-datasets/PhysioCGM/<subject_id>.csv`
6. To calculate the participant-level metadata, run the following:

```bash
python PhysioCGM_metadata.py Standardized-datasets/PhysioCGM
```

