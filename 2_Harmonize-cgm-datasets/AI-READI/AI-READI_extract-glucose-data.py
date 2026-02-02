import pandas as pd
import sys
import os
import json
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_aireadi_data(json_path, subject_id, dst):
    """
    Processes AI-READI JSON data files and extracts blood glucose measurements.
    
    This function reads JSON files from the AI-READI dataset, extracts the continuous glucose 
    monitoring (CGM) data, normalizes it into a pandas DataFrame, selects relevant columns,
    and saves the processed data as CSV files.
    
    Args:
        root (str): Path to the root directory containing subject folders with JSON files
        dst (str): Path to destination directory where processed CSV files will be saved
    
    Returns:
        None: The function saves CSV files to the specified destination directory
    """
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    cgm_data = data["body"]["cgm"]

    df = pd.json_normalize(cgm_data)


    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df = df[
        [
            "effective_time_frame.time_interval.start_date_time",
            "blood_glucose.value",
        ]
    ].rename(
        columns={
            "effective_time_frame.time_interval.start_date_time": "timestamp",
            "blood_glucose.value": "glucose_value_mg_dl",
        }
    )


    df["timestamp"] = (pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%SZ").dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    df.to_csv(dst / f"{subject_id}.csv", index=False)

def main():
    """
    Main function that parses command-line arguments and initiates data processing.
    
    Expects two command-line arguments:
    1. Path to input folder containing AI-READI data
    2. Path to output folder where processed CSV files will be saved
    
    If incorrect number of arguments is provided, displays usage instructions and exits.
    
    Returns:
        None
    """
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python AI-READI_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    #Create output directory "Standardized-datasets" to store processed CSV file outputs.
    output_dir = "Standardized-datasets/AI-READI"
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    # Loop through raw directory contents and pull the subject ID from the raw file name.
    for subject in input_path.rglob("**/*_DEX.json"):
        subject_id = subject.parent.stem#pulls subjectID to use for output file generation.
        clean_aireadi_data(subject, subject_id, output_dir)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')


if __name__ == "__main__":
    main()