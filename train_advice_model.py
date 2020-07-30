from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import image_processors
import models

NUM_EPOCHS = 3
NUM_FRAMES = 1 # todo: i'm not sure if frames still work for training. may depend on model, check outputs.
DATASET_PATH = "SuperMarioBros-v3_data"
OUTPUT_PATH = "mario-test.model"

class AdviceDataset(Dataset):
    def __init__(self, dataset_path):
        # metadata on all of the images
        self.images_csv = pd.read_csv(dataset_path + "/img_data.csv")
        # the root directory containing the images
        self.root_dir = dataset_path + "/images"

        poss_actions = self.images_csv['action'].unique()
        labels_df = pd.DataFrame()
        for action in poss_actions:
            labels_df[str(action)] = np.where(self.images_csv['action'] == action, np.float32(1), np.float32(0))

        self.num_possible_actions = len(poss_actions)

        # todo: it might be good to have a self.y_names field that is an array that corresponds to the correct actions
        # for example, in mario it would be [0, 1, 2, 3, 4, 5, 6]. it can be useful if there is an environment
        # where the labels don't always start up at 0 and always match the argmax

        # sort the y columns in numerical order
        labels_df = labels_df.reindex(sorted(labels_df.columns), axis=1).astype('float')
        # convert y to a tensor
        self.y = torch.tensor(labels_df.to_numpy())

        # TODO: dealing with image height and width is awkward. I think i need to call my processor once here to derive the value
        # self.img_height =
        # self.img_width =

    def __len__(self):
        return len(self.images_csv)

    def __getitem__(self, idx):
        # todo: do i even want this istensor if? i think i get rid.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = []
        for i in range(NUM_FRAMES - 1, -1, -1):
            if idx - i < 0:
                j = 0
            else:
                j = i
            img_name = os.path.join(self.root_dir, self.images_csv.iloc[idx-j, 0])
            image = Image.open(img_name)
            imgs.append(image_processors.downsample(image))

        # the current x shape is (num_frames???, 1 (grayscale only), height, width)
        # the current y shape is (num_possible_actions,)
        return torch.cat(imgs, 0), self.y[idx].reshape(self.num_possible_actions,)

adv_dataset = AdviceDataset(DATASET_PATH)

model = getattr(models, "AdviceModel")()

# sample the imbalanced data so that everything is weighted evenly ( 1 / weights )
weights = adv_dataset.y.numpy().sum(axis=0)
weights_balanced = 1. / weights
actions = adv_dataset.images_csv['action']
samples_weight = torch.tensor(np.array(weights_balanced)[actions])
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# todo: look into training with > 1 batch_size
dataloader = DataLoader(adv_dataset, batch_size=1, sampler=sampler)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# main training loop
for epoch in range(NUM_EPOCHS):
    print("epoch " + str(epoch))
    running_loss = 0
    misses = 0
    hits = 0
    # this is actually wrong, since we miss out on num_frames - 1 / num_frames training samples...
    for index, data in enumerate(dataloader):
        x, y_true = data
        # get the model's current prediction
        y_pred = model(x)

        # compute the loss with CrossEntropyLoss, which takes the argmax of the true label
        loss = loss_fn(y_pred, torch.argmax(y_true).reshape((1,)))

        # update the weights based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate some data
        if torch.argmax(y_pred).item() == torch.argmax(y_true).item():
            hits += 1
        else:
            misses += 1
        running_loss += loss.item()

        # at the end of the epoch, print some stats
        if index == len(adv_dataset) - 1:
            print("Total Loss: " + str(running_loss))
            print("Misses: " + str(misses) + ", Hits: " + str(hits))

torch.save(model.state_dict(), OUTPUT_PATH)

# todo: put that if name == main thing to call the function