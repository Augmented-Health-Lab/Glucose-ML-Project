import subprocess
import argparse
from pathlib import Path
import sys


LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def dataset_library(arg):
    datasets = {"hall_2018": "Hall_2018",
            "d1namo": "D1NAMO",
            "colas_2019": "Colas_2019",
            "ohiot1dm": "OhioT1DM",
            "t1dexi": "T1DEXI",
            "t1dexip": "T1DEXIP",
            "bigideas": "BIGIDEAs",
            "diatrend": "DiaTrend",
            "shanghait1dm": "ShanghaiT1DM", 
            "shanghait2dm": "ShanghaiT2DM",
            "t1diabetesgranada": "T1DiabetesGranada", 
            "ai-readi": "AI-READI",
            "uchtt1dm": "UCHTT1DM",
            "hupa-ucm": "HUPA-UCM",
            "cgmacros_dexcom": "CGMacros_Dexcom",
            "cgmacros_libre": "CGMacros_Libre",
            "t1d-uom": "T1D-UOM",
            "bris-t1d_open": "Bris-T1D_Open",
            "azt1d": "AZT1D",
            "park_2025": "Park_2025",
            "physiocgm": "PhysioCGM"
            }
    return datasets[arg]


def standardize_datasets(arg):
    
    dataset_string = dataset_library(arg)

    print(f"{LIME_GREEN}Glucose-ML{R}: Harmonizing the {LIGHT_RED}{dataset_string}{R} dataset.")

    #base_dir points to ../Glucose-ML/Auto-scripts
    base_dir = Path(__file__).resolve().parent
    #harmonize_dir points to ../Glucose-ML/harmonize-CGM-datasets/Bris-T1D_Open
    harmonize_dir = base_dir.parent / "harmonize-CGM-datasets" / dataset_string
    #Handels dataset downloads that contain more than 1 Glucose-ML dataset and splits them,
    if dataset_string == "CGMacros_Dexcom" or dataset_string == "CGMacros_Libre":
        raw_data_path = (base_dir / "Original-Glucose-ML-datasets" / f"CGMacros_raw_data")
    elif dataset_string == "ShanghaiT1DM":
        raw_data_path = (base_dir / "Original-Glucose-ML-datasets" / f"Shanghai_raw_data" / "diabetes_datasets" / "Shanghai_T1DM")
    elif dataset_string == "ShanghaiT2DM":
        raw_data_path = (base_dir / "Original-Glucose-ML-datasets" / f"Shanghai_raw_data" / "diabetes_datasets" / "Shanghai_T2DM")
    else:
        raw_data_path = (base_dir / "Original-Glucose-ML-datasets" / f"{dataset_string}_raw_data")
    #raw_data_path = Path(f"Original-Glucose-ML-datasets/{dataset_string}_raw_data")
    call_script_1 = harmonize_dir / f"{dataset_string}_extract-glucose-data.py"
    call_script_2 = harmonize_dir / f"{dataset_string}_metadata.py"

    call_script_2_input = f"Standardized-datasets/{dataset_string}"

    meta_output_path = Path(f"Standardized-metadata")
    meta_output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Run script 1
        subprocess.run([sys.executable, str(call_script_1), str(raw_data_path)],check=True)
        # Run script 2 but only if script 1 goes through.
        subprocess.run([sys.executable, str(call_script_2), call_script_2_input],check=True)

    except subprocess.CalledProcessError as e:
        print(f"{LIGHT_RED}Glucose-ML{R}: Error while processing {dataset_string}: {e}")





def main():

    #standardize_datasets("hall_2018")
    parser = argparse.ArgumentParser(description="This script standardizes a Glucose-ML-friendly datasets & generates some metadata. Dataset options: ")
    parser.add_argument("datasets", nargs="+", type=str, help="Specify the dataset(s) to standardize. Speparate datasets with spaces if standardizing more than 1.")  # Initializes 'datasets' Argument.

    input_args = parser.parse_args()

    for arg in input_args.datasets:
        try:
            standardize_datasets(arg)
        except KeyError:
            print(f"{LIGHT_RED}Glucose-ML{R}: Unknown dataset provided: {arg}")
        except Exception as e:
            print(f"{LIGHT_RED}Glucose-ML{R}: Failed to standardize the following dataset {arg}: {e}")

if __name__ == "__main__":
    main()