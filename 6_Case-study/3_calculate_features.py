from pathlib import Path
import pandas as pd
import numpy as np


def calculate_features(participant_df, participant, project, participant_pop, split_assignment):
    """
    """
    glucose_reads = pd.to_numeric(participant_df["glucose_value_mg_dl"], errors="coerce").dropna().reset_index(drop=True)
    feature_bin = {
        "person_id": participant,
        "dataset": project,
        "diabetes_type": participant_pop,
        "split_assignment": split_assignment,
        "adrr": np.nan,
        "mage": np.nan,
        "sddm": np.nan,
        "n_readings": len(glucose_reads),
        "mean_glucose": glucose_reads.mean(),
        "n_days_present": np.nan,
        "bgri": np.nan, 
        "pct_below_70": (glucose_reads < 70).sum() / len(glucose_reads) * 100,
        "pct_in_70_140": ((glucose_reads >= 70) & (glucose_reads <= 140)).sum() / len(glucose_reads) * 100,
        "pct_in_70_180": ((glucose_reads >= 70) & (glucose_reads <= float(180))).sum() / len(glucose_reads) * 100,
        "pct_in_180_250": ((glucose_reads > 180) & (glucose_reads <= 250)).sum() / len(glucose_reads) * 100,
        "pct_above_250": (glucose_reads > 250).sum() / len(glucose_reads) * 100,
        "hbgi": np.nan, 
        "max_glucose": glucose_reads.max(),
        "avg_readings_per_day": np.nan,
        "min_glucose": glucose_reads.min(),
        "range_glucose": glucose_reads.max() - glucose_reads.min(),
        "median_glucose": glucose_reads.median(),
        "cv_glucose": np.nan,
        "sdw": np.nan,
        "lbgi": np.nan,
        "j_index": np.nan,
        "sd_glucose": glucose_reads.std(ddof=1),
        "iqr_glucose": glucose_reads.quantile(0.75) - glucose_reads.quantile(0.25),
    }

    #cv_glucose
    if feature_bin["mean_glucose"] > 0 and feature_bin["sd_glucose"] >0:
        feature_bin["cv_glucose"] = (feature_bin["sd_glucose"] / feature_bin["mean_glucose"]) * 100
    
    # MAGE
    mage = calculate_mage(glucose_reads)
    if mage > 0:
        feature_bin["mage"] = mage

    # Calculate J-index
    j_index = 0.001 * (glucose_reads.mean() + glucose_reads.std(ddof=1)) ** 2
    if j_index > 0:
        feature_bin["j_index"] = j_index

    # Calculate LBGI, HBGI, BGRI
    glucose_reads_over_0 = glucose_reads[glucose_reads >0]
    intermediate_function = 1.509 * ((np.log(glucose_reads_over_0) ** 1.084) - 5.381)
    risk_value = 10 * (intermediate_function ** 2)

    low_risk = risk_value[intermediate_function < 0]
    high_risk = risk_value[intermediate_function > 0]


    if len(low_risk) > 0:
        feature_bin["lbgi"] = low_risk.mean()
    else:
        feature_bin["lbgi"] = 0

    if len(high_risk) > 0:
        feature_bin["hbgi"] = high_risk.mean()
    else:
        feature_bin["hbgi"] = 0

    feature_bin["bgri"] = feature_bin["lbgi"] + feature_bin["hbgi"]

    # Calculate n_days_present and avg_readings_per_day
    if "timestamp" in participant_df.columns:
        temp_df = participant_df.copy()
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], errors="coerce")
        temp_df = temp_df.dropna(subset=["timestamp"])

        if len(temp_df) > 0:
            temp_df["date"] = temp_df["timestamp"].dt.date
            reads_per_day = temp_df.groupby("date").size()

            feature_bin["n_days_present"] = len(reads_per_day)

            if len(reads_per_day) > 0:
                feature_bin["avg_readings_per_day"] = reads_per_day.mean()

            # Calculate sddm (SD of daily means)
            daily_means = temp_df.groupby("date")["glucose_value_mg_dl"].mean()
            if len(daily_means) >= 2:
                feature_bin["sddm"] = daily_means.std(ddof=1)

            # Calculate sdw (mean of daily SDs)
            daily_sds = temp_df.groupby("date")["glucose_value_mg_dl"].std(ddof=1).dropna()
            if len(daily_sds) > 0:
                feature_bin["sdw"] = daily_sds.mean()

            # Calculate ADRR
            daily_risk_ranges = []

            for day, day_df in temp_df.groupby("date"):
                day_glucose = pd.to_numeric(day_df["glucose_value_mg_dl"], errors="coerce").dropna()
                day_glucose = day_glucose[day_glucose > 0]

                if len(day_glucose) == 0:
                    continue

                f = 1.509 * ((np.log(day_glucose) ** 1.084) - 5.381)

                low_risks = 10 * (f[f < 0] ** 2)
                high_risks = 10 * (f[f > 0] ** 2)

                lr = low_risks.max() if len(low_risks) > 0 else 0
                hr = high_risks.max() if len(high_risks) > 0 else 0

                daily_risk_ranges.append(lr + hr)

            if len(daily_risk_ranges) > 0:
                feature_bin["adrr"] = np.mean(daily_risk_ranges)
    
    
    return feature_bin

def calculate_mage(glucose_reads):
    """
    Calculate MAGE (Mean Amplitude of Glycemic Excursions)
    """

    if len(glucose_reads) < 3:
        return np.nan

    # Standard deviation threshold
    sd = glucose_reads.std(ddof=1)

    # Find peaks and troughs manually
    turning_points = []

    for i in range(1, len(glucose_reads) - 1):
        prev_val = glucose_reads[i - 1]
        curr_val = glucose_reads[i]
        next_val = glucose_reads[i + 1]

        # Peak
        if curr_val > prev_val and curr_val > next_val:
            turning_points.append(curr_val)
        # Trough
        elif curr_val < prev_val and curr_val < next_val:
            turning_points.append(curr_val)

    if len(turning_points) < 2:
        return np.nan
    
    # Compute excursions
    excursions = []
    for i in range(1, len(turning_points)):
        diff = abs(turning_points[i] - turning_points[i - 1])
        if diff > sd:
            excursions.append(diff)

    if len(excursions) == 0:
        return np.nan

    return np.mean(excursions)

def main():
    '''
    Calculate features to use for Machine Learning Case Study.

    Inputs:
        - "Processed-Data/preprocessing_manifest.csv"
        - "Processed-Data/[dataset]/[subject].csv"
    Output: 
        - "feature_calcs.csv"

    '''

    manifest = pd.read_csv("Processed-Data/preprocessing_manifest.csv", dtype={"person_id": str})
    manifest = manifest[manifest["passed"] == "yes"] # Only pull participants who passed.
    project_ids = manifest["dataset"].unique() # Pull dataset ids.

    final_df = []
    for project in project_ids:
        project_df = manifest[manifest["dataset"]==project]
        for i, entry in project_df.iterrows():
            participant = entry["person_id"]
            participant_file_path = Path("Processed-Data") / project / f'{participant}.csv'
            participant_file = Path(participant_file_path)
            participant_pop = entry["diabetes_type"]
            split_assignment = entry["split_assignment"]
            participant_df = pd.read_csv(participant_file)
            participant_features = calculate_features(participant_df, participant, project, participant_pop, split_assignment)
            final_df.append(participant_features)

    
    features_df = pd.DataFrame(final_df)
    features_df.to_csv("feature_calcs.csv", index=False)


    print("Done!")

if __name__ == "__main__":
    main()