import os
import pandas as pd
import sys
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_t1diabetesgranada_data(input_file, output_folder):
    '''
    Cleans and standardizes T1DiabetesGranada CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Process each Patient_ID
    count = 0
    for patient_id, patient_data in df.groupby('Patient_ID'):
        try:
            # Create the "timestamp" column by combining "Measurement_date" and "Measurement_time"
            patient_data['timestamp'] = patient_data['Measurement_date'] + ' ' + patient_data['Measurement_time']

            # Rename the "Measurement" column to "glucose_value_mg_dl"
            patient_data = patient_data.rename(columns={'Measurement': 'glucose_value_mg_dl'})

            # Keep only the required columns
            patient_data = patient_data[['timestamp', 'glucose_value_mg_dl']]

            # Drop rows missing timestamps or glucose values
            patient_data = patient_data.dropna(subset=["timestamp", "glucose_value_mg_dl"])

            # Ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)

            # Construct the output file path
            output_file = os.path.join(output_folder, f"{patient_id}.csv")

            # Save the processed data to a new CSV file
            patient_data.to_csv(output_file, index=False)
            count += 1
        except Exception as e:
            print(f"{LIGHT_RED}Glucose-ML{R}: Error processing data for Patient_ID {patient_id}: {e}")
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')
def main():
    '''
    Processes raw data from the T1DiabetesGranada dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory of .xml files.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python T1DiabetesGranada_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that the raw data csv exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    #Create output directory "Standardized-datasets" to store processed CSV file outputs.
    rglob_raw_data = list(input_path.glob("*cose_measurements.csv"))
    if len(rglob_raw_data) == 0:
        raise FileNotFoundError(f"{LIGHT_RED}Glucose-ML{R}: Error - No LB.csv found under: {input_path}")
    if len(rglob_raw_data) > 1:
        raise RuntimeError(f"{LIGHT_RED}Glucose-ML{R}: Error - Multiple LB.csv files found: {rglob_raw_data}")


    output_dir = "Standardized-datasets/T1DiabetesGranada"
    os.makedirs(output_dir, exist_ok=True)

    clean_t1diabetesgranada_data(rglob_raw_data[0], output_dir)

if __name__ == "__main__":
    main()