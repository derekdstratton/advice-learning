from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np

class AdviceDataset(Dataset):
    def __init__(self):
        self.images_csv = pd.read_csv('Breakout-v0_data/img_data.csv')
        self.root_dir = 'Breakout-v0_data/images'

    def __len__(self):
        return len(self.images_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images_csv.iloc[idx, 0])
        image = Image.open(img_name)
        arr = np.array(image.convert('L'))
        arr.resize((210, 160, 1))
        return torch.tensor(arr)
        # return np.array(image.convert('L'))

adv_dataset = AdviceDataset()

print(adv_dataset[0].shape)
height = adv_dataset[0].shape[0]
width = adv_dataset[0].shape[1]
conv_dim = (height - 3 + 1) * (width - 3 + 1) * 5

model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.Linear(conv_dim, 3),
)

# todo: i need a DataLoader...

print(model(adv_dataset[0]))