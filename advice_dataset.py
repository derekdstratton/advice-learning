from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import image_processors

class AdviceDataset(Dataset):
    def __init__(self, dataset_path, num_frames, IMG_PROCESSOR):
        # metadata on all of the images
        self.images_csv = pd.read_csv(dataset_path + "/img_data.csv")
        # the root directory containing the images
        self.root_dir = dataset_path + "/images"

        poss_actions = self.images_csv['action'].unique()
        labels_df = pd.DataFrame()
        for action in poss_actions:
            labels_df[str(action)] = np.where(self.images_csv['action'] == action, np.float32(1), np.float32(0))

        self.num_possible_actions = len(poss_actions)
        self.num_frames = num_frames

        # todo: it might be good to have a self.y_names field that is an array that corresponds to the correct actions
        # for example, in mario it would be [0, 1, 2, 3, 4, 5, 6]. it can be useful if there is an environment
        # where the labels don't always start up at 0 and always match the argmax

        # sort the y columns in numerical order
        labels_df = labels_df.reindex(sorted(labels_df.columns), axis=1).astype('float')
        # convert y to a tensor
        self.y = torch.tensor(labels_df.to_numpy())

        img_name = os.path.join(self.root_dir, self.images_csv.iloc[0, 0])
        image = Image.open(img_name)
        self.processor = getattr(image_processors, IMG_PROCESSOR)
        sample_image = self.processor(image)
        self.img_height = sample_image.shape[1]
        self.img_width = sample_image.shape[2]

    def __len__(self):
        return len(self.images_csv)

    def __getitem__(self, idx):
        # todo: do i even want this istensor if? i think i get rid.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = []

        # lets you gather multiple frames going back
        for i in range(self.num_frames - 1, -1, -1):
            if idx - i < 0:
                j = 0
            else:
                j = i
            img_name = os.path.join(self.root_dir, self.images_csv.iloc[idx-j, 0])
            image = Image.open(img_name)
            imgs.append(self.processor(image))

        # the current x shape is (num_frames???, 1 (grayscale only), height, width)
        # the current y shape is (num_possible_actions,)
        return torch.cat(imgs, 0), self.y[idx].reshape(self.num_possible_actions,)