<p align="center">
  <img src="Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="500">
</p>

<h1 align="center"> Glucose-ML: An evolving collection of continuous glucose datasets to accelerate data-centric AI for diabetes</h1>


## Abstract
Wearable continuous glucose monitors (CGMs) collect large volumes of personal health data that is critical to develop transformative solutions for diabetes prevention and care. However, limited access to high-quality datasets is a significant barrier that impedes progress for data-centric researchers and innovators. To address this challenge, we present Glucose-ML -- an evolving collection of publicly available continuous glucose datasets -- curated to push the boundaries of data science and machine research, and accelerate development of next-generation computing solutions. At present, Glucose-ML comprises 44.9 million continuous glucose samples from 20+ datasets collected from 4,400+ individuals across 6 countries. This unique resource spans populations with type 1 diabetes (38\%), type 2 diabetes (25\%), prediabetes (13\%), and no diabetes (24\%). To support researchers and innovators with leveraging multiple datasets from the Glucose-ML collection, we provide automated tools for seamlessly downloading and harmonizing continuous glucose data from each dataset and the associated metadata (e.g., demographic information, clinical metrics, and CGM features). Additionally, we provide an interactive web interface (https://glucose-ml-project.com/) to support dataset exploration, visualization, and selection for researchers and practitioners. Guided by our learnings from curating and consolidating the Glucose-ML collection, we also provide insights and recommendations for dataset creators and curators to support future research and development. Our provided dataset resources and related code are openly accessible here: (https://github.com/Augmented-Health-Lab/Glucose-ML-Project)

## Citing Glucose-ML

Please cite the associated paper as follows:
- **Prioleau, T., Lu, B. and Cui, Y., 2026. Glucose-ML: An evolving collection of continuous glucose datasets to accelerate data-centric AI for diabetes.**


## Overview

This repository contains the code developed for harmonized analysis of 20+ publicly available diabetes datasets curated in the Glucose-ML collection as published here: [insert publication here]. This repository does _not_ host any of the datasets directly, but _instead_ provides easy-to-use resources for downloading and standardizing compatible datasets. 

To support ease of use, this repository provides automated scripts in the ```1_Auto-scripts/``` directory for downloading the open-access diabetes datasets (14) in the Glucose-ML collection, and automated scripts (20) for harmonizing and jointly analyzing all 20 publicly available datasets, including metadata for all participants with accessible CGM data (open-access datasets only).

## Getting Started

1. To download the Glucose-ML software input the following into your command line to clone this repository:

```bash
git clone https://github.com/Augmented-Health-Lab/Glucose-ML-Project.git
```

2. Glucose-ML requires the installation of 3 dependencies to function: pandas, requests, and numpy. To install all required dependencies in one download, input the following:

```bash
pip install -r dependencies.txt
```


## Contents & Structure

1. ```1_Auto-scripts/```
    Contains the 2 main scripts for downloading and processing Glucose-ML compatible datasets:
    - ```auto-download-open-datasets.sh``` : Automatically download 1, 2, or all 12 open-access datasets.
    - ```auto-harmonize-CGM-datasets.sh``` : Automatically standardize 1, 2, or all 20 open-access AND controlled-access raw dataset downloads & calculate metadata all at once.
    
    *For more information and detailed directions on using these scripts please refer to the instructions in [README](/1_Auto-scripts/README.md).

2. ```2_Harmonized-cgm-datasets/```
    Contains helper scripts that work directly with ```auto-harmonize-CGM-datasets.sh``` that standardize and calcualte metadata for each Glucose-ML compatible project.
    - ```{Project-ID}/```
        - ```{Project-ID}_extract-glucose-data.py```
        - ```{Project-ID}_metadata.py```

    *For more information regarding these scripts please refer to the instructions in [README](/2_Harmonize-cgm-datasets/README.md).

3. ```3_Glucose-ML-collection/```
    Includes pre-standardized datasets, metadata, or a README depending on the dataset accessibility.

    - Open-access Datasets: 
        - Contains standardized CGM data for all participants with valid CGM data avalible and pulled metadata for each of these participants.

    - Controlled-access Datasets:
        - Each project-level folder contains a README that includes information on where to request access for the dataset & information on how you can use ```auto-harmonize-CGM-datasets.sh``` to standardize the raw data if aquired. 

4. ```4_Figures-from-paper/```
    Includes all Figures from the official publication.

5. ```5_Tables-from-paper/```
    Includes all Tables from the official publication.


## Run Automation Scripts
To reproduce the Glucose-ML pipeline (from data acquisition to data standardization) use the scripts located in the ```1_Auto-scripts/``` directory. They can be run as follows:

1. Downloading Datasets with ```auto-download-open-datasets.sh``` (dataset compatibility: open-accessonly)

    ```bash
    cd 1_Auto-scripts/
    python3 auto-download-open-datasets.py {dataset 1} {dataset 2} ... {dataset n}
    ```

    **Compatible datasets:** azt1d, bigideas, bris-t1d_open, cgmacros, d1namo, hupa-ucm, park_2025, physiocgm, shanghai, t1d-uom, uchtt1dm

    _Note_: colas_2019 and hall_2018 are open-access datasets but can only be downloaded through their respective publications.

    For more detailed information please see [README](/1_Auto-scripts/README.md)

2. Harmonizing Datasets with ```auto-harmonize-CGM-datasets.sh``` (dataset compatibility: open and controlled)
    ```bash
    cd 1_Auto-scripts/
    python3 auto-harmonize-CGM-datasets.sh.py {dataset 1} {dataset 2} ... {dataset n}
    ```
    **Compatible datasets:** ai-readi, azt1d, bigideas, bris-t1d_open, cgmacros_dexcom, cgmacros_libre, colas_2019, d1namo, diatrend, hupa-ucm, ohiot1dm, park_2025, physiocgm, shanghait1dm, shanghait2dm, hall_2018, colas_2019, ohiot1dm, t1dexi, t1dexip, t1diabetesgranada, t1d-uom, uchtt1dm

    For more detailed information please see [README](/1_Auto-scripts/README.md) 

## Comparative Analysis

 ```4_Figures-from-paper``` and ```5_Tables-from-paper``` includes the figures and tables presented in the paper, respectively. For the code to plot those figures and the tables included in the paper, please refer to ```Scripts/``` found in both directories.

## Project Contributors
- Ryan Pontius
- Worayada Pitakanonda
- Zimo Li
- Kultum Lhabaik
- Fengran Wang
- Baiying Lu
- Yanjun Cui
- Temiloluwa Prioleau

## Questions, Comments or Feedback

Please reach out to the Principal Investigator: Temiloluwa Prioleau, PhD ([tpriole@emory.edu](mailto:tpriole@emory.edu)).
Should I add my info too??

## License

This project is licensed under the MIT License.
