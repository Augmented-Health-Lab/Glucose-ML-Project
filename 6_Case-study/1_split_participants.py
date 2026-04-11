from pathlib import Path
import pandas as pd





def split_data(df,seed,split_proportions):
    split_groups = []

    diabetes_pops = df["diabetes_type"].unique()

    for diabetes_status in diabetes_pops:
        group = df[df["diabetes_type"] == diabetes_status].copy()

        # Randomly shuffle rows in this group using seed.
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split the data
        sample_count = len(group)
        train_n = round(sample_count * split_proportions["train"])
        val_n = round(sample_count * split_proportions["validate"])
        test_n = sample_count - train_n - val_n

        #Handle small groups so negatives dont occur.
        if test_n < 0:
            test_n = 0
            total = train_n + val_n
            overflow = total - sample_count

            if overflow > 0:
                if train_n >= overflow:
                    train_n -= overflow
                else:
                    overflow -= train_n
                    train_n = 0
                    val_n = max(0, val_n - overflow)

        split_labels = []

        # Assignment
        for i in range(train_n):
            split_labels.append("train")

        for i in range(val_n):
            split_labels.append("validate")

        for i in range(test_n):
            split_labels.append("test")

        group["split_category"] = split_labels
        split_groups.append(group)

    #Regroup split dfs
    split_df = pd.concat(split_groups).reset_index(drop=True)
    return split_df



def main():
    '''
    This script accesses the metadata for open-access datasets and splits the data by Diabetes status and
    applies a random test-train-validate split on each diabetes population.

    Inputs: 3_Glucose-ML-collection/[project]-metadata.csv
    Output: participant_splits.csv
    '''

    seed = 20
    script_path = Path(__file__).resolve()
    glucose_ml_dir = script_path.parent.parent
    split_proportions = {"test": 0.2, "train": 0.7, "validate": 0.1}
    summ = 0
    for value in split_proportions.values():
        summ += value

    if round(summ, 8) != 1.0:
        raise ValueError("Error: split proportions must equal 1!")
        
    open_projects = ["AZT1D", "BIGIDEAs", "Bris-T1D_Open", "CGMacros_Dexcom", "Colas_2019", "D1NAMO", "Hall_2018", "HUPA-UCM", 'PhysioCGM', "ShanghaiT1DM", "ShanghaiT2DM", "T1D-UOM", "UCHTT1DM"]
    
    final_df = []
    #Iterate through open project metadata
    for project in open_projects:
        project_path = f'{project}-metadata.csv'
        metadata_path = glucose_ml_dir / "3_Glucose-ML-collection" / project / project_path
        metadata = Path(metadata_path)
        df = pd.read_csv(metadata,dtype={"person_id": str})

        df = df[["person_id", "diabetes_type"]]
        df_assignments = split_data(df, seed, split_proportions)
        df_assignments["dataset"] = project
        final_df.append(df_assignments)

    final_df = pd.concat(final_df).reset_index(drop=True)
    final_df.to_csv("participant_splits.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()