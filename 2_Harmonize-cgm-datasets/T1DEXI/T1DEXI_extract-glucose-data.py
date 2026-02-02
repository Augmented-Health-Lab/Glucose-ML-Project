import os
import pandas as pd
import sys
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_t1dexi_data(df, output_dir):
    '''
    Cleans and standardizes T1DEXI CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # Convert subject id's to integers.
    df['USUBJID'] = df['USUBJID'].astype(int)

    # sort rows by subject id.
    df = df.sort_values("USUBJID")

    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"LBDTC": "timestamp", "LBORRES": "glucose_value_mg_dl"}, inplace=True)

    # # Create a df that contains one row per subject, regardless if they have HB1AC data or not.
    all_subjects = pd.DataFrame({"USUBJID": df["USUBJID"].unique()}) 

    #Create a new df that only contains glucose readings.
    df = df[df["LBTESTCD"] != "HBA1C"].copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    # Loop through each subject...
    count = 0
    for subj in all_subjects["USUBJID"]:
        # Create new df for the subject with timestamp & glucose data.
        subj_df = df[df["USUBJID"] == subj][["timestamp", "glucose_value_mg_dl"]]

        # Doesn't make a csv file if participant doesnt have any glucose data.
        if subj_df.empty:
            continue

        # Create output file for the subject.
        filename = os.path.join(output_dir, f"{subj}.csv")
        subj_df.to_csv(filename, index=False)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')


def main():
    '''
    Processes raw data from the T1DEXI dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory.
    Outputs: 1) Standardized CSV files for each subject. Creates a directory "Standardized-datasets" that will contain the generated output.
             2) "hb1ac_per_subject.csv" - All HB1AC readings for each subject.

    Each subject standardized csv output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python T1DEXI_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data csv.
    input_path = Path(sys.argv[1])

    # Path to directory containing the raw data CSV.
    rglob_raw_data = list(input_path.rglob("**/LB.csv"))
    if len(rglob_raw_data) == 0:
        raise FileNotFoundError(f"{LIGHT_RED}Glucose-ML{R}: Error - No LB.csv found under: {input_path}")
    if len(rglob_raw_data) > 1:
        raise RuntimeError(f"{LIGHT_RED}Glucose-ML{R}: Error - Multiple LB.csv files found: {rglob_raw_data}")
    
    rglob_path = rglob_raw_data[0]
    raw_data_file = pd.read_csv(rglob_path)

    #Create output directory "Standardized-datasets" to store standardized CSV file outputs.
    output_dir = "Standardized-datasets/T1DEXI"
    os.makedirs(output_dir, exist_ok=True)

    clean_t1dexi_data(raw_data_file, output_dir)


if __name__ == "__main__":
    main()