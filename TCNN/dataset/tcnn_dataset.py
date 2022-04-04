import os
import random

import librosa
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from util.utils import synthesis_noisy_y,sliceframe


class TCNNDataset(Dataset):
    def __init__(self,
                 noise_dataset="/home/imucs/Datasets/Build-SE-Dataset-V2/Data/noise.txt",
                 clean_dataset="/home/imucs/Datasets/Build-SE-Dataset-V2/Data/clean.txt",
                 snr_list=None,
                 offset=700,
                 limit=None,
                 mode="train",
                 n_jobs=-1
                 ):
       
        super().__init__()
        assert mode in ["train", "validation", "test"], "mode parameter must be one of 'train', 'validation', and 'test'."
        clean_f_paths = [line.rstrip('\n') for line in open(clean_dataset, "r")]
        clean_f_paths = clean_f_paths[offset:]
        if limit:
            clean_f_paths = clean_f_paths[:limit]

        noise_f_paths = [line.rstrip('\n') for line in open(noise_dataset, "r")]

        def load_noise_file(file_path, sr=16000):
            basename_text = os.path.basename(os.path.splitext(file_path)[0])
            y, _ = librosa.load(file_path, sr=sr)
            return {
                "name": basename_text,
                "y": y
            }

        all_noise_data = Parallel(n_jobs=n_jobs)(delayed(load_noise_file)(f_path, sr=16000) for f_path in tqdm(noise_f_paths, desc=f"Loading {mode} noise files"))

        self.length = len(clean_f_paths)
        self.all_noise_data = all_noise_data
        self.clean_f_paths = clean_f_paths
        self.snr_list = snr_list
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean_y, _ = librosa.load(self.clean_f_paths[idx], sr=16000)
        snr = random.choice(self.snr_list)

        noise_data = random.choice(self.all_noise_data)
        noise_name = noise_data["name"]
        noise_y = noise_data["y"]

        name = f"{str(idx).zfill(5)}_{noise_name}_{snr}"
        clean_y, noise_y, noisy_y = synthesis_noisy_y(clean_y, noise_y, snr)

        if self.mode == "train":
            
            noisy_mag = sliceframe(noisy_y)
            clean_mag = sliceframe(clean_y)
            n_frames = clean_mag.shape[-1]
            
            return noisy_mag, clean_mag, n_frames
        elif self.mode == "validation":
            return noisy_y, clean_y, name
        else:
            return noisy_y, name


if __name__ == '__main__':
    dataset = TCNNDataset(snr_list=["-5", "-4", "-3", "-2", "-1", "0", "1"])
    res = next(iter(dataset))
    print(res[0].shape)
    print(res[1].shape)
    print(res[2])
