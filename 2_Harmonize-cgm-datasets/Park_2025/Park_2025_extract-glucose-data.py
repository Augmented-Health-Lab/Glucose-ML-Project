import os
import pandas as pd
import sys
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_park_2025_data(df, output_dir):
    '''
    Cleans and standardizes Park_2025 CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"mins_since_start": "timestamp", "glucose": "glucose_value_mg_dl"}, inplace=True)

    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    # Loop to generate csv output files for each subject.
    count = 0
    for subj in df["subject"].unique():
        subj_df = df[df["subject"] == subj][["timestamp", "glucose_value_mg_dl"]]
        filename = os.path.join(output_dir, f"{subj}.csv")
        subj_df.to_csv(filename, index=False)
        #print(f"Saved file for subject {subj}: {filename}")
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')


def main():
    '''
    Processes raw data from the Park_2025 dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python Park_2025_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    # Raw CSV file input to extract data from.
    csv_files = list(input_path.glob("*.csv"))
    raw_data_file = pd.read_csv(csv_files[0])

    #Create output directory to store CSV file outputs.
    output_dir = "Standardized-datasets/Park_2025"
    os.makedirs(output_dir, exist_ok=True)


    clean_park_2025_data(raw_data_file, output_dir)

if __name__ == "__main__":
    main()