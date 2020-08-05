from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import image_processors
import models
import advice_dataset
import os
import json

# todo: use pathlib instead of os ideally

NUM_EPOCHS = 1
NUM_FRAMES = 1
DATASET_PATH = "SuperMarioBros-v3_data"
MODEL = "AdviceModel"
GAME = "SuperMarioBros-v3" # technically not used, except for in test_advice

# in the future, these can be user configurable
OUTPUT_MAIN_DIRECTORY = "models"
if not os.path.exists(OUTPUT_MAIN_DIRECTORY):
    os.makedirs(OUTPUT_MAIN_DIRECTORY)

OUTPUT_SECOND_DIRECTORY = OUTPUT_MAIN_DIRECTORY + "/" + GAME + "_" + MODEL


OUTPUT_SECOND_DIRECTORY_tmp = OUTPUT_SECOND_DIRECTORY
index = 0
while (os.path.isdir(OUTPUT_SECOND_DIRECTORY_tmp)):
    OUTPUT_SECOND_DIRECTORY_tmp = OUTPUT_SECOND_DIRECTORY + "_" + str(index)
    index += 1
os.mkdir(OUTPUT_SECOND_DIRECTORY_tmp)

OUTPUT_MODEL_PATH = OUTPUT_SECOND_DIRECTORY_tmp + "/model.pt"
OUTPUT_INFO_PATH = OUTPUT_SECOND_DIRECTORY_tmp + "/info.json"

IMG_PROCESSOR = "downsample" # todo: make an option to not use any (None)

info = {
    "num_frames": NUM_FRAMES,
    "num_epochs": NUM_EPOCHS,
    "game": GAME,
    "dataset_path": DATASET_PATH,
    "model": MODEL,
    "img_processor": IMG_PROCESSOR
}

adv_dataset = advice_dataset.AdviceDataset(DATASET_PATH, NUM_FRAMES, IMG_PROCESSOR)

model = getattr(models, MODEL)(adv_dataset.img_height, adv_dataset.img_width, adv_dataset.num_possible_actions, NUM_FRAMES)

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

        # print(y_pred) # debugging

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

torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
fp = open(OUTPUT_INFO_PATH, "w")
json.dump(info, fp) # todo: i could use the := here to be fancy, and use with open() as
fp.close()
print("Files output to: " + OUTPUT_SECOND_DIRECTORY_tmp)

# todo: put that if name == main thing to call the function