# 1_Auto-scripts - auto-download-open-datasets.py & auto-harmonize-CGM-datasets.py

This directory includes two command-line scripts that automate the primary Glucose-ML pipeline:

1) Download supported open datasets into a consistent `Original-Glucose-ML-datasets/` structure.  
2) Run dataset-specific harmonization scripts to create standardized outputs and metadata.

---

## What’s included?

### 1. `auto-download-open-datasets.py`
Downloads one or more datasets (by short name), saves them under `Original-Glucose-ML-datasets/`, and automatically unzips archives when needed (only `.zip` files are unpacked).

* Dependencies:
  * requests (python -m pip install requests)
  * pandas (pip install pandas)

Key features:
- Accepts multiple datasets in one command.
- Asks for confirmation after estimating total download size.
- Streams downloads and prints periodic progress updates.
- Unzips `.zip` downloads automatically (if appropriate).

_NOTE_: Only open-access datasets may be downloaded using the auto-download-open-datasets.py script. See which datasets can be downloaded in the "Download script keys" section below.

### 2. `auto-harmonize-CGM-datasets.py`
Takes one or more datasetkeys and runs a two-step harmonization pipeline by calling dataset-specific scripts in `auto-harmonize-CGM-datasets/`:

1) `<Dataset>_extract-glucose-data.py` (fed the raw dataset directory)  
2) `<Dataset>_metadata.py` (input is `Standardized-datasets/<Dataset>`)

Key features:
- Accepts multiple datasets (open or controlled access) in one command.
- Standardzizes raw dataset downloads and calculates metadata statistics.

_NOTE_: The auto-harmoniza-CGM-datasets.py script can standardize datasets that cannot be downloaded from the auto-download-open-datasets.py script so long as they are part of the 20 Glucose-ML datasets (See "Harmonize script keys" section below for compatible datasets.)

It also handles cases where one download contains multiple datasets (e.g. CGMacros Dexcom/Libre and Shanghai T1DM/T2DM) by pointing each harmonizer at the correct subfolder.

---

## Folder layout (expected)

These scripts assume a project layout like:

- `Auto-scripts/`
  - `auto-download-open-datasets.py`
  - `auto-harmonize-CGM-datasets.py`
  - `Original-Glucose-ML-datasets/` (created by auto-download-open-datasets.py)
- `harmonize-CGM-datasets/`
  - `<DatasetName>/`
    - `<DatasetName>_extract-glucose-data.py`
    - `<DatasetName>_metadata.py`
- `Standardized-datasets/` (created & used by harmonization dataset-specific scripts)
- `Standardized-metadata/` (created by harmonize wrapper)

---

## Quick start

### 1) Downloading dataset(s)



To download any number of datasets, run:

```bash
cd Glucose-ML/Auto-scripts
python auto-download-open-datasets.py d1namo bigideas cgmacros 
```

You’ll see an estimated total download size (raw zipped + unzipped) and be asked to confirm before anything downloads. Downloads are written to:

- `Original-Glucose-ML-datasets/<Dataset>_raw_data/`

If a dataset download is a `.zip`, it will be unpacked into the same folder automatically.

---

### 2) Harmonize (standardize) dataset(s)
After downloading or manually adding the unzipped raw dataset, run:

```bash
cd Glucose-ML/Auto-scripts
python auto-harmonize-CGM-datasets.py d1namo bigideas cgmacros_dexcom shanghait1dm
```

For each dataset, the script:
- Computes the correct raw-data directory.
- Runs the dataset’s extract script, then its metadata script.
- Ensures `Standardized-metadata/` exists.

---

## Harmonizing Controlled-Access Datasets
Though not all Glucose-ML datasets can be downloaded using the auto-download script, all Glucose-ML datasets can be standardized using the `auto-harmonize-CGM-datasets.py` if set up properly.

Steps to add a controlled-access dataset:
1. Create or locate the directory `Auto-scripts/Original-Glucose-ML-datasets/`
2. Create a folder with the following nomenclature `Auto-scripts/Original-Glucose-ML-datasets/<Dataset>_raw_data/`
3. Rename `<Dataset>_raw_data/` to the proper dataset. This can be done by copying/pasteing the appropriate dataset key from the "Harmonize script keys" section below. (e.g. `Auto-scripts/Original-Glucose-ML-datasets/ai-readi_raw_data/`)
4. Add the download into the `<Dataset>_raw_data/` directory exactly how it was downloaded.
5. Unzip the download if it needs to be unzipped (don't do anything else at this point.)
6. Execute the `auto-harmonize-CGM-datasets.py` script as described above.


## Supported dataset keys

### Download script keys
These are the dataset keys recognized by `auto-download-open-datasets.py` (only open-access datasets can be downloaded):

- `azt1d`
- `bigideas`
- `bris-t1d_open`
- `cgmacros`
- `d1namo`
- `hupa-ucm`
- `park_2025`
- `physiocgm`
- `shanghai` (download bundle used for both T1DM and T2DM)
- `t1d-uom`
- `uchtt1dm`


Special Cases:
- `cgmacros` downloads BOTH `cgmacros_libre` and `cgmacros_dexcom`.
- `shanghai` downloads BOTH `shanghait1dm`, `shanghait2dm`.

_NOTE_: `colas_2019` and `hall_2018` are open-access datasets but can only be downloaded through their respective publications. 

### Harmonize script keys
`auto-harmonize-CGM-datasets.py`:
- `ai-readi`, `azt1d`, `bigideas`, `bris-t1d_open`,
- `cgmacros_dexcom`, `cgmacros_libre`, `colas_2019`, `d1namo`,
- `diatrend`, `hupa-ucm`, `ohiot1dm`, `park_2025`,
- `physiocgm`, `shanghait1dm`, `shanghait2dm`,
- `hall_2018`, `colas_2019`, `ohiot1dm`, `t1dexi`,
- `t1dexip`, `t1diabetesgranada`, `t1d-uom`, `uchtt1dm`

---

## Examples

### Example 1
Download + harmonize a Shanghai split:
```bash
python auto-download-open-datasets.py shanghait1dm
python auto-harmonize-CGM-datasets.py shanghait1dm
```

Even though you requested `shanghait1dm`, the downloader grabs the shared Shanghai zip. The harmonizer then points specifically to:

- `Original-Glucose-ML-datasets/Shanghai_raw_data/diabetes_datasets/Shanghai_T1DM`

### Example 2
Download once, harmonize both CGMacros device sources:
```bash
python auto-download-open-datasets.py cgmacros
python auto-harmonize-CGM-datasets.py cgmacros_dexcom cgmacros_dexcom
```

The download script saves everything under `Original-Glucose-ML-datasets/CGMacros_raw_data/`. The harmonizer script reuses that same raw path for both Dexcom and Libre harmonization jobs.

---

## Troubleshooting

- **“Unknown dataset”**  
  The dataset key isn’t in the script’s internal mapping. Check the supported keys above (and note the download and harmonize scripts don’t have identical lists).

- **Download progress looks wrong**  
  Progress percent is computed using the expected file size from the raw, unzipped file only. If the upstream file changes size, percentages may drift.

---

## Suggested workflow

1) Download everything you need:
```bash
python auto-download-open-datasets.py d1namo bigideas bris-t1d_open
```

2) Harmonize what you downloaded:
```bash
python auto-harmonize-CGM-datasets.py t1diabetesgranada bris-t1d_open park_2025 ai-readi
```

---

<p>&nbsp;</p>

<p align="center">
  <img src="../Logos/glucose-ml-logo_stacked.svg" alt="Glucose-ML logo" width="450">
</p>