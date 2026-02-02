import os
import pandas as pd
from pathlib import Path
import sys
import shutil

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_azt1d_data(df, subject_id, output_dir):
    '''
    Cleans and standardizes AZT1D CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # This dataset has various column names for glucose readings. This conditional accounts for these differences.
    if "Readings (CGM / BGM)" in df.columns:
        # Rename columns & convert timestamp data to the standardized names used throughout the project.
        df.rename(columns={"EventDateTime": "timestamp", "Readings (CGM / BGM)": "glucose_value_mg_dl"}, inplace=True)
    else:
        # Rename columns & convert timestamp data to the standardized names used throughout the project.
        df.rename(columns={"EventDateTime": "timestamp", "CGM": "glucose_value_mg_dl"}, inplace=True)

    #Convert timestamp column to Pandas readable format.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    subj_df = df[["timestamp", "glucose_value_mg_dl"]]

    # Create an output csv for each subject using the subj_df variable.
    outfile = os.path.join(output_dir, f"{subject_id}.csv")
    subj_df.to_csv(outfile, index=False)


def main():
    '''
    Processes raw data from the AZT1D dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python AZT1D_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])
    source_data_path = Path(input_path)


    #Create output directory "Standardized-datasets" to store CSV file outputs.
    output_dir = "Standardized-datasets/AZT1D"
    os.makedirs(output_dir, exist_ok=True)

    for zip_path in source_data_path.rglob("*2025.zip"):
        extract_dir = zip_path.parent
        print(f"Unzipping: {zip_path}")
        shutil.unpack_archive(zip_path, extract_dir)

    # Loop through raw directory contents and pull the subject ID from the raw file name.
    count = 0
    for subject in input_path.rglob("*Diabetes/AZT1D 2025/CGM Records/**/**/*.csv"):
        print(subject)
        df=pd.read_csv(subject)
        subject_id = subject.parent.name #pull the subject ID from the raw file name.
        clean_azt1d_data(df, subject_id, output_dir)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')

if __name__ == "__main__":
    main()