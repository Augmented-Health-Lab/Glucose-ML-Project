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

1. Download the PhysioCGM dataset from the original [data source](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Furldefense.com%2Fv3%2F__https%3A%2Fdoi.org%2F10.6084%2Fm9.figshare.28136294__%3B!!KwNVnqRv!CpauS_FLCRXjpklB38zaqPu5r-5xuhhMBWfQJHIcCa72dESFv3xiQYT5deUXMMGrhVu3Yfc56cPr%24&data=05%7C02%7Ctemiloluwa.prioleau%40emory.edu%7Ca101cc20bc7544f6d0b108de28622579%7Ce004fb9cb0a4424fbcd0322606d5df38%7C0%7C0%7C638992599205738451%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=syyGO0gRORqKBm%2BhyUHn5Uct3XPl%2Bm5%2B9RTJPLWeAhU%3D&reserved=0) or use our auto_download scripts [here](../../1_Auto-scripts). If the download is a zipped file, DO NOT UNZIP.
2. Place the unzipped download in `1_Auto-scripts/Original-Glucose-ML-datasets/PhysioCGM_raw_data/` (Make the directory, if needed.)
3. Execute the following:
```bash
cd 2_Harmonize-cgm-datasets/PhysioCGM
python PhysioCGM_extract-glucose-data.py ../../1_Auto-scripts/Original-Glucose-ML-datasets/PhysioCGM_raw_data/
```
4. Harmonized csv files are written to: `Standardized-datasets/PhysioCGM/<subject_id>.csv`
5. To calculate the participant-level metadata, run the following:

```bash
python PhysioCGM_metadata.py Standardized-datasets/PhysioCGM
```

