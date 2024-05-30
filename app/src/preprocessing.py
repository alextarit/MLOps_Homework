# Import libraries
import pandas as pd
import numpy as np

# Selecting the columns to delete
drop_cols = ["client_id", "mrg_", "регион", "pack",  "зона_1", "использование", "зона_2", "pack_freq"]

def import_data(path_to_file):

    #Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=drop_cols)

    return input_df

def run_preproc(input_df):

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    print('Train data imported...')

    if "доход" in input_df.columns:
        input_df["доход"] = input_df["доход"].round(-2)

    numeric_cols = input_df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if col != "доход":
            input_df[col] = input_df[col].round(3)

    print("Preprocessing complete...")

    return input_df  