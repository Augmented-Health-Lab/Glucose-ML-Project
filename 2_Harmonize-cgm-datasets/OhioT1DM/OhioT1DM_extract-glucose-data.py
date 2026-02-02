import os
import xml.etree.ElementTree as ET
import csv
from datetime import datetime
import sys
from pathlib import Path

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_ohiot1dm_data(input_folder, output_folder):
    '''
    Cleans and standardizes OhioT1DM CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''
    # Get all XML file names in the input folders 2018/2020.
    subject_rows = {}
    subject_counts = {}

    for root_dir, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith(".xml"):
                continue

            xml_path = os.path.join(root_dir, file)

            try:
                # Parse XML file into an element tree
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Assign patient id using the id column.
                patient_id = root.attrib["id"]

                # Initialize data structures for this subject if not seen before
                subject_rows.setdefault(patient_id, [])
                subject_counts.setdefault(patient_id, {"total": 0, "written": 0})
                # Find the CGM data.
                glucose_node = root.find("glucose_level")
                # Skipts non-existent glucose data.
                if glucose_node is None:
                    continue
                # Iterate over all CGM events.
                for event in glucose_node.findall("event"):
                    subject_counts[patient_id]["total"] += 1
                    try:
                        # Extract raw timestamp and glucose value from attributes
                        ts = event.attrib["ts"]
                        value = event.attrib["value"]
                        # Pull and Format the collection date.
                        formatted_ts = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")

                        subject_rows[patient_id].append((formatted_ts, value))
                        subject_counts[patient_id]["written"] += 1
                    except Exception:
                        pass

            except Exception as e:
                print(f"{LIGHT_RED}Glucose-ML{R}: Error processing {xml_path}: {e}")
    
    # Write merged CSVs (one per subject)
    count = 0
    for patient_id, rows in subject_rows.items():
        # Sort records chronologically by datetime
        rows.sort(key=lambda x: x[0])

        output_csv = os.path.join(output_folder, f"{patient_id}.csv")
        # Writes the standardized csv output.
        with open(output_csv, mode="w", newline="") as csv_file:
            count += 1
            writer = csv.writer(csv_file)
            # Standardized output column names.
            writer.writerow(["timestamp", "glucose_value_mg_dl"])
            for ts, value in rows:
                writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), value])

    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')
def main():
    '''
    Processes raw data from the OhioT1DM dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory of .xml files.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python OhioT1DM_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_folder = Path(sys.argv[1])

    #Create output directory "Standardized-datasets" to store processed CSV file outputs.
    output_dir = "Standardized-datasets/OhioT1DM"
    os.makedirs(output_dir, exist_ok=True)

    clean_ohiot1dm_data(input_folder, output_dir)

if __name__ == "__main__":
    main()