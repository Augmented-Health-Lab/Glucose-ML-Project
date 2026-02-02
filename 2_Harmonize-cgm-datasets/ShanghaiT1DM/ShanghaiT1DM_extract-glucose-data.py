import pandas as pd
import sys
import os
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def determine_engine(filename):
    """
    Determine which Excel engine to use based on file extension
    """
    if filename.lower().endswith('.xlsx'):
        return 'openpyxl'
    elif filename.lower().endswith('.xls'):
        return 'xlrd'
    else:
        # Default to openpyxl for unknown extensions
        return 'openpyxl'

def clean_shanghait1dm_data(root, dst):
    """
    Cleans and standardizes ShanghaiT1DM CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    
    Args:
        root (str): Path to the directory containing Shanghai T1DM Excel files
        dst (str): Path to destination directory where processed CSV files will be saved
    
    Returns:
        None: The function saves processed CSV files to the specified destination directory
    """

    subj_dict = {}
    # Iterrate through raw data files.
    for file in os.listdir(root):
        if file.endswith('.xlsx') or file.endswith('.xls'):
            if file.split('_')[0] not in subj_dict:
                subj_dict.update({file.split('_')[0]: [file]})
            else:
                subj_dict[file.split('_')[0]].append(file)
    count = 0
    for subj in subj_dict.keys():
        count += 1
        if len(subj_dict[subj]) == 1:
            file_path = os.path.join(root, subj_dict[subj][0])
            # Determines file root
            engine = determine_engine(subj_dict[subj][0])
            try:
                df = pd.read_excel(file_path, sheet_name=subj_dict[subj][0].split('.')[0], engine=engine)
                # Rename columns & convert timestamp data to the standardized names used throughout the project.
                df_selected = df[['Date', 'CGM (mg / dl)']].rename(columns={'Date': 'timestamp', 'CGM (mg / dl)': 'glucose_value_mg_dl'})
                # Drop rows missing timestamps or glucose values
                df_selected = df_selected.dropna(subset=["timestamp", "glucose_value_mg_dl"])

                df_selected.to_csv(os.path.join(dst, subj+'.csv'), index=None)
            except Exception as e:
                print(f"{LIGHT_RED}Glucose-ML{R}: Error processing {file_path}: {e}")
        # subject with multiple files
        else: 
            subj_dict[subj].sort()
            df_list = []
            for file in subj_dict[subj]:
                file_path = os.path.join(root, file)
                # Determines file root
                engine = determine_engine(file)
                try:
                    df = pd.read_excel(file_path, sheet_name=file.split('.')[0], engine=engine)
                    df_list.append(df)
                except Exception as e:
                    print(f"{LIGHT_RED}Glucose-ML{R}: Error processing {file_path}: {e}")
            
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                # Rename columns & convert timestamp data to the standardized names used throughout the project.

                df_selected = df[['Date', 'CGM (mg / dl)']].rename(columns={'Date': 'timestamp', 'CGM (mg / dl)': 'glucose_value_mg_dl'})
                # Drop rows missing timestamps or glucose values
                df_selected = df_selected.dropna(subset=["timestamp", "glucose_value_mg_dl"])

                df_selected.to_csv(os.path.join(dst, subj+'.csv'), index=None)
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')
    
    
def main():
    '''
    Processes raw data from the ShanghaiT1DM dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory of .xml files.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python ShanghaiT1DM_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    #Create output directory "Standardized-datasets" to store processed CSV file outputs.
    output_dir = "Standardized-datasets/ShanghaiT1DM"
    os.makedirs(output_dir, exist_ok=True)
    
    clean_shanghait1dm_data(input_path, output_dir)

if __name__ == "__main__":
    main()