import pandas as pd
import sys
import os
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_shanghait2dm_data(root, dst):
    """
    Cleans and standardizes ShanghaiT2DM CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    
    Args:
        root (str): Path to the directory containing Shanghai T2DM Excel files
        dst (str): Path to destination directory where processed CSV files will be saved
    
    Returns:
        None: The function saves processed CSV files to the specified destination directory
    """

    subj_dict = {}
    for file in os.listdir(root):
        if file.split('_')[0] not in subj_dict:
            subj_dict.update({file.split('_')[0]: [file]})
        else:
            subj_dict[file.split('_')[0]].append(file)
    count = 0
    for subj in subj_dict.keys():
        if len(subj_dict[subj]) == 1: # subject only has one record
            df = pd.read_excel(os.path.join(root, subj_dict[subj][0]))
        else: # subject with multiple files
            subj_dict[subj].sort() # sorted by time
            df_list = [pd.read_excel(os.path.join(root,file)) for file in subj_dict[subj]]
            df = pd.concat(df_list, ignore_index=True)

        try:
            df_selected = df[['Date', 'CGM (mg / dl)']].rename(columns={'Date': 'timestamp', 'CGM (mg / dl)': 'glucose_value_mg_dl'})
        except: # subject 2045
            df_selected = df[['Date', 'CGM ']].rename(columns={'Date': 'timestamp', 'CGM ': 'glucose_value_mg_dl'})

        # Drop rows missing timestamps or glucose values
        df_selected = df_selected.dropna(subset=["timestamp", "glucose_value_mg_dl"])
        df_selected.to_csv(os.path.join(dst, subj+'.csv'), index=None)
        count += 1
            # break
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')

def main():
    """
    Main function that parses command-line arguments and initiates data processing.
    
    Expects two command-line arguments:
    1. Path to input folder containing Shanghai T2DM Excel files
    2. Path to output folder where processed CSV files will be saved
    
    If incorrect number of arguments is provided, displays usage instructions and exits.
    
    Returns:
        None
    """
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python ShanghaiT1DM_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = sys.argv[1]
    source_data_path = Path(input_path)

    #Create output directory "Standardized-datasets" to store processed CSV file outputs.
    output_dir = "Standardized-datasets/ShanghaiT2DM"
    os.makedirs(output_dir, exist_ok=True)
    
    clean_shanghait2dm_data(source_data_path, output_dir)

if __name__ == "__main__":
    main()