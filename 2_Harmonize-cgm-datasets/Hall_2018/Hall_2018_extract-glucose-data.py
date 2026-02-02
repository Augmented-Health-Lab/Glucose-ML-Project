import os
import pandas as pd
import sys
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_hall_2018_data(df, output_dir):
    '''
    Cleans and standardizes Hall_2018 CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''
    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"DisplayTime": "timestamp", "GlucoseValue": "glucose_value_mg_dl"}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors="raise")

    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    # Loop to generate csv output files for each subject.
    count = 0
    for subj in df["subjectId"].unique():
        subj_df = df[df["subjectId"] == subj][["timestamp", "glucose_value_mg_dl"]]
        filename = os.path.join(output_dir, f"{subj}.csv")
        subj_df.to_csv(filename, index=False)
        count +=1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')


def main():
    '''
    Processes raw data from the Hall_2018 dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''
    
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python Hall_2018_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw UNZIPPED download.
    input_path = Path(sys.argv[1])
    data_file = next(input_path.glob("pbio.*.s*"))

    #Create output directory to store standardized CSV file outputs.
    output_dir = "Standardized-datasets/Hall_2018"
    os.makedirs(output_dir, exist_ok=True)
    
    raw_data_file = pd.read_csv(data_file, sep="\t")
    clean_hall_2018_data(raw_data_file, output_dir)



if __name__ == "__main__":
    main()