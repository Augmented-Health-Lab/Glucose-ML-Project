import os
import pandas as pd
from pathlib import Path
import sys

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_d1namo_data(df, subject_id, output_dir):
    '''
    Cleans and standardizes D1NAMO CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format with additional quality checks.
    - Writing per-subject CSV files containing timestamped glucose values
    - Convert glucose records from raw mmol/L units to the project-standard mg/dL units.
    '''

    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"glucose": "glucose_value_mg_dl"}, inplace=True)

    # Standardize time stamp data.
    #print(df["date"].iloc[0], df["time"].iloc[0])
    # Need to handle different timestamp formats, so push to function parse_timestamp to figure out the inconsistencies.
    df["timestamp"] = parse_timestamp(df["date"] + " " + df["time"])
    
    df["glucose_value_mg_dl"] = pd.to_numeric(df["glucose_value_mg_dl"],errors="coerce")

    # Convert glucose records from mmol/L to mg/dL
    df["glucose_value_mg_dl"] = (df["glucose_value_mg_dl"] * 18).round(1)
    
    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    subj_df = df[["timestamp", "glucose_value_mg_dl"]]

    #Make blank csv output file.
    outfile = os.path.join(output_dir, f"{subject_id}.csv")
    # Populate the output file
    subj_df.to_csv(outfile, index=False)

def parse_timestamp(input):
    """
    Handle timestamps with various formats. This dataset has logged glucose readings in day/time formats, 
    so to standardize safely this function attempts to parse the timestamps using various ISO-formatting types.
    """

    #Try parsing the timestamps using the most common format.
    timestamp = pd.to_datetime(input,format="%Y-%m-%d %H:%M",errors="coerce")

    # If first parse fails, try parsing again on the failed rows using the other format.
    mask = timestamp.isna()
    if mask.any():
        timestamp.loc[mask] = pd.to_datetime(input[mask], format="mixed", errors="raise")

    return timestamp
        
def main():
    '''
    Processes raw data from the D1NAMO dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    
    Input: Raw data directory.
    Output: Standardized CSV files for each subject. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each subject output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    # NOTE: There are 2 seprate raw data directories for the D1NAMO dataset, one for the diabetes cohort and 
    # one for the healthy cohort.

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python D1NAMO_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])


    #Create output directory "Standardized-datasets" to store standardized CSV file outputs.
    output_dir = "Standardized-datasets/D1NAMO"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all the files that are named "glucose.csv", as this is where the glucose readings are stored.
    sourcedata_files = input_path.rglob("diabetes_subset*/*/glucose.csv")
    count = 0
    for subject in sourcedata_files:
        df=pd.read_csv(subject)
        subject_id = subject.parent.name
        clean_d1namo_data(df, subject_id, output_dir)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')

if __name__ == "__main__":
    main()