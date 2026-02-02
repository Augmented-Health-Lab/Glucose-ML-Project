import os
import pandas as pd
from pathlib import Path
import sys

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_colas_2019_data(df, subject_id, output_dir):
    '''
    Cleans and standardizes Colas_2019 CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"hora": "timestamp", "glucemia": "glucose_value_mg_dl"}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S").dt.time

    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    # Collect timestamp & glucose readings into a new dataframe. 
    subj_df = df[["timestamp", "glucose_value_mg_dl"]]

    # Create an output csv for each subject using the subj_df variable.
    outfile = os.path.join(output_dir, f"{subject_id}.csv")
    subj_df.to_csv(outfile, index=False)


def main():
    '''
    Processes raw data from the Colas_2019 dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory.
    Output: Standardized CSV files for each subject. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each subject output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python Colas_2019_extract-demographics.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to where the Colas_2019_extract-glucose-data.py output lives.
    input_path = Path(sys.argv[1])
    
    #Create output directory "Standardized-datasets" to store standardized CSV file outputs.
    output_dir = "Standardized-datasets/Colas_2019"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through raw directory contents and pull the subject ID from the raw file name.
    count = 0
    for subject in input_path.rglob("**/*.csv"):
        subject_id = subject.stem #pulls subjectID to use for output file generation.
        df=pd.read_csv(subject)
        clean_colas_2019_data(df, subject_id, output_dir)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')



if __name__ == "__main__":
    main()