# src/dataset.py

import os
import glob
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset


class SagT2Dataset(Dataset):
    def __init__(self, df, label_cols, img_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_middle_slice(self, study_id, series_id):
        series_path = os.path.join(self.img_root, str(study_id), str(series_id))
        dcm_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))

        if len(dcm_files) == 0:
            raise RuntimeError(f"No DICOM files found in {series_path}")

        mid_idx = len(dcm_files) // 2
        dcm = pydicom.dcmread(dcm_files[mid_idx])
        img = dcm.pixel_array.astype(np.float32)

        # normalize
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        study_id = row["study_id"]
        series_id = row["series_id_sag_t2"]

        img = self.load_middle_slice(study_id, series_id)

        if self.transform:
            img = self.transform(img)

        # convert 1-channel → 3-channel
        img = img.repeat(3, 1, 1)

        labels = torch.tensor(row[self.label_cols].values, dtype=torch.long)

        return img, labels