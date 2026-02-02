import os
import pandas as pd
import sys
from pathlib import Path

def clean_physiocgm_data(df, subject_id, output_dir):
    '''
    Cleans and standardizes PhysioCGM CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format with additional quality checks.
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"Timestamp (YYYY-MM-DDThh:mm:ss)": "timestamp", "Glucose Value (mg/dL)": "glucose_value_mg_dl"}, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('s')#Need to round because Excel stores datetimes as floating-point numbers
    first_row_of_data = df["glucose_value_mg_dl"].first_valid_index()
    if first_row_of_data is not None:
        df = df.loc[first_row_of_data:]

    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    subj_df = df[["timestamp", "glucose_value_mg_dl"]]

    #Make blank csv output file.
    outfile = os.path.join(output_dir, f"{subject_id}.csv")
    # Populate the output file
    subj_df.to_csv(outfile, index=False)


def main():
    '''
    Processes raw data from the PhysioCGM dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    
    Input: Raw data directory.
    Output: Standardized CSV files for each subject. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each subject output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python PhysioCGM_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    #Create output directory "Standardized-datasets" to store CSV file outputs.
    output_dir = "Standardized-datasets/PhysioCGM"
    os.makedirs(output_dir, exist_ok=True)

    # Find all the files that contain the glucose readings.

    sourcedata_files = input_path.rglob("**/*_raw/cgm.csv")
    for subject in sourcedata_files:
        print(subject)
        df=pd.read_csv(subject)
        subject_id = subject.parent.name.replace("_raw", "")
        clean_physiocgm_data(df, subject_id, output_dir)

if __name__ == "__main__":
    main()