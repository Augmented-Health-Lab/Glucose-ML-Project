import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

def plot_gender(data_raw):
    '''
    This function generates a png of Figure 3a (Population breakdown of Sex/Gender)
    '''
    plt.figure()
    sex_gender = data_raw["Sex/Gender"].str.split(" / ")
    male_count= 0
    female_count = 0
    unknown_count = 0
    for counts in sex_gender:
        male_count += int(counts[0])
        female_count += int(counts[1])
        unknown_count += int(counts[2])


    bars = plt.bar(["Male", "Female"], [male_count, female_count], width = 0.5, color=["#B489F0"])
    plt.ylim(0, 4000)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,f"{int(height)}",ha="center",va="bottom", fontsize=21)

    plt.ylabel("# Participants",fontsize=23, labelpad=10)
    plt.xlabel("Sex/Gender",fontsize=23, labelpad=10)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig("Figures/Figure_3a.png", dpi=600)

def plot_diabetes_type(base_directory):
    '''
    This function generates a png of Figure 3d (Population breakdown of Diabetes Type)
    '''
    plt.figure()

    t1d = 0
    t2d = 0
    prediabetic = 0
    no_diabetes = 0
    unknown = 0

    for metadata_file in base_directory.rglob("*metadata.csv"):
        if metadata_file.name == "CGMacros_Libre-metadata.csv":
            continue
        metadata_df = pd.read_csv(metadata_file)
        diabetes_df = metadata_df["diabetes_type"]
        for i in diabetes_df:
            if pd.isna(i):
                unknown += 1
                continue
            i = i.lower()
            if i == "t2d":
                t2d += 1
            elif i == "t1d":
                t1d += 1
            elif i == "no diabetes":
                no_diabetes += 1
            elif i == "prediabetes":
                prediabetic += 1
            else:
                unknown += 1


    bars = plt.bar(["T1D", "T2D", "preD", "ND"], [t1d, t2d, prediabetic, no_diabetes], width = 0.8, color=["#91EFDE"])
    plt.ylim(0, 4000)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,f"{int(height)}",ha="center",va="bottom", fontsize=21)

    plt.ylabel("# Participants",fontsize=23, labelpad=10)
    plt.xlabel("Diabetes Type",fontsize=23, labelpad=10)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig("Figures/Figure_3d.png", dpi=600)

def plot_age(base_directory):
    '''
    This function generates a png of Figure 3b (Population breakdown of Age groups)
    '''
    plt.figure()
    bin_less_than_18 = 0
    bin_18_34 = 0
    bin_35_50 = 0
    bin_51_69 = 0
    bin_70_plus = 0
    invalid_ages = 0
    errors = []
    for metadata_file in base_directory.rglob("*metadata.csv"):
        if metadata_file.name == "CGMacros_Libre-metadata.csv":
            continue
        metadata_df = pd.read_csv(metadata_file)
        age_df = pd.to_numeric(metadata_df["age"], errors="coerce")
        dropped_age_count = age_df.isna().sum()
        invalid_ages += dropped_age_count
        age_df = age_df.dropna()

        for age in age_df:
            if age < 18:
                bin_less_than_18 += 1
            elif age >= 18 and age < 35:
                bin_18_34 += 1
            elif age >= 35 and age < 51:
                bin_35_50 += 1 
            elif age >= 51 and age < 70:
                bin_51_69 += 1 
            elif age >= 70:
                bin_70_plus += 1
            else:
                errors.append(age)
    
    
    
    bars = plt.bar(["< 18", "[18-34)", "[34-50)", "[50-69)", "70+"], [bin_less_than_18, bin_18_34, bin_35_50, bin_51_69, bin_70_plus], width = 0.9, color=["#F276AD"])
    plt.ylim(0, 4000)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,f"{int(height)}",ha="center",va="bottom", fontsize=21)
    plt.ylabel("# Participants",fontsize=23, labelpad=10)
    plt.xlabel("Age (years)",fontsize=23, labelpad=10)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig("Figures/Figure_3b.png", dpi=600)


def plot_race_eth(base_directory):
    '''
    This function generates a png of Figure 3c (Population breakdown of Race/Ethnicity)
    '''
    plt.figure()
    reported = 0
    not_reported = 0
    for metadata_file in base_directory.rglob("*metadata.csv"):
        if metadata_file.name == "CGMacros_Libre-metadata.csv":
            continue
        metadata_df = pd.read_csv(metadata_file)
        race_df = metadata_df["race_ethnicity"]
        for i in race_df:
            if pd.isna(i) or str(i) == "":
                not_reported += 1
            else:
                reported += 1
    bars = plt.bar(["Reported", "Not Reported"], [reported, not_reported], width = 0.5, color = ["#EFA1C8"])
    plt.ylim(0, 4000)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,f"{int(height)}",ha="center",va="bottom", fontsize=21)
    plt.ylabel("# Participants",fontsize=23, labelpad=10)
    plt.xlabel("Race/Ethnicity",fontsize=23, labelpad=10)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig("Figures/Figure_3c.png", dpi=600)

def main():
    # Table 1 csv.
    raw_csv = Path("../5_Tables-from-paper/Tables/Table_3.csv")
    data_raw = pd.read_csv(raw_csv)
    data_raw = data_raw.drop([0, 15]) # Drop one of the CGMacros rows.

    #Path to metadata files
    #base_directory = Path("/Users/ryanpontius/Desktop/AugmentedHealth/Glucose-ML/Glucose-ML-collection")
    base_directory = Path("../3_Glucose-ML-collection")
    #Output directory creation.
    outdir = Path("Figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_gender(data_raw)
    plot_diabetes_type(base_directory)
    plot_age(base_directory)
    plot_race_eth(base_directory)

if __name__ == "__main__":
    main()