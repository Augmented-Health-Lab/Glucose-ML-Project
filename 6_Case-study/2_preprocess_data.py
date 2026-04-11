from pathlib import Path
import pandas as pd
import numpy as np
import re


def process_files(project_df, extracted_glucose_file_path, project, max_days, minimum_coverage):
    manifest_rows = []

    for i, row in project_df.iterrows():
        person_id = str(row["person_id"])
        diabetes_type = str(row["diabetes_type"])
        split_category = str(row["split_category"])
        dataset = str(row["dataset"])

        person_data = extracted_glucose_file_path / f"{person_id}.csv"

        if not person_data.exists():
            if not "c2s02.csv":
                raise ValueError(f"{person_data} does not exist!")

        try:
            df = pd.read_csv(person_data)

            df = df[["timestamp", "glucose_value_mg_dl"]].copy()

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["glucose_value_mg_dl"] = pd.to_numeric(df["glucose_value_mg_dl"], errors="coerce")

            n_rows_raw = len(df)

            if n_rows_raw == 0:
                manifest_rows.append({
                    "person_id": person_id,
                    "diabetes_type": diabetes_type,
                    "split_category": split_category,
                    "dataset": dataset,
                    "status": "no_valid_rows_after_cleaning",
                    "n_rows_raw": 0,
                    "n_rows_processed": 0,
                    "n_valid_days": 0
                })
                continue

            df = df.set_index("timestamp")
            df = df.resample("5min").median()

            # 15 minute max interpolation gap at (assumes 5 min samp rate)
            df["glucose_value_mg_dl"] = df["glucose_value_mg_dl"].interpolate(method="time", limit=3, limit_direction="both")

            df = df.reset_index()
            df["date"] = df["timestamp"].dt.date

            day_counts = (df.groupby("date")["glucose_value_mg_dl"].count().reset_index())
            day_counts.columns = ["date", "non_missing_count"]
            day_counts["coverage"] = day_counts["non_missing_count"] / 288

            valid_days = []

            for j, day_row in day_counts.iterrows():
                if day_row["coverage"] >= minimum_coverage:
                    valid_days.append(day_row["date"])

            valid_days = valid_days[:max_days]

            df = df[df["date"].isin(valid_days)].copy()
            df = df.drop(columns=["date"])

            n_rows_processed = len(df)
            n_valid_days = len(valid_days)

            out_folder = Path("Processed-Data") / dataset
            out_folder.mkdir(parents=True, exist_ok=True)

            output_file = out_folder / f"{person_id}.csv"

            if n_rows_processed == 0:
                status = "no_valid_days_after_preprocessing"
            else:
                df.to_csv(output_file, index=False)
                status = "ok"

            manifest_rows.append({
                "person_id": person_id,
                "diabetes_type": diabetes_type,
                "split_category": split_category,
                "dataset": dataset,
                "status": status,
                "num_rows_raw": n_rows_raw,
                "num_rows_processed": n_rows_processed,
                "num_valid_days": n_valid_days
            })

        except Exception as e:
            manifest_rows.append({
                "person_id": person_id,
                "diabetes_type": diabetes_type,
                "split_category": split_category,
                "dataset": dataset,
                "status": f"error: {e}",
                "num_rows_raw": np.nan,
                "num_rows_processed": np.nan,
                "num_valid_days": np.nan
            })

    return manifest_rows




def main():
    max_days = 30
    minimum_coverage = 0.7

    script_path = Path(__file__).resolve()
    glucose_ml_dir = script_path.parent.parent
    

    split_participants_df = pd.read_csv("participant_splits.csv",dtype={"person_id": str})
    project_ids = split_participants_df["dataset"].unique()
    
    final_rows = []

    for project in project_ids:
        project_df = split_participants_df[split_participants_df["dataset"]==project]

        project_path = f'{project}-extracted-glucose-files'
        extracted_glucose_file_path = glucose_ml_dir / "3_Glucose-ML-collection" / project/ project_path

        project_rows = process_files(project_df, extracted_glucose_file_path, project, max_days, minimum_coverage)

        for row in project_rows:
            final_rows.append(row)

    manifest_df = pd.DataFrame(final_rows)
    manifest_df.to_csv("Processed-Data/preprocessing_manifest.csv", index=False)

    print("Done!")

if __name__ == "__main__":
    main()