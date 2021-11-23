import os
from pathlib import Path
import pandas as pd

BRATS_TRAIN_FOLDERS = f"../../MICCAI_BraTS2020_TrainingData"
# BRATS_VAL_FOLDER = f"../../MICCAI_BraTS2020_ValidationData/"
# we are not using this cause we don't have the ground truth on these


def get_train_val_df(n_train, n_val=None):
    # n_train: on train_count number of patients

    naming = pd.read_csv(f'{BRATS_TRAIN_FOLDERS}/name_mapping.csv')
    data_df = pd.DataFrame(naming['BraTS_2020_subject_ID'])

    train_df = data_df[:n_train]
    if n_val:
        valid_df = data_df[n_train: n_train + n_val]
    else:
        valid_df = data_df[n_train:]

    return train_df, valid_df
