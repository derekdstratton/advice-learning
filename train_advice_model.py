import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np

num_frames = 1
num_possible_actions = 7

class AdviceDataset(Dataset):
    def __init__(self):
        self.images_csv = pd.read_csv('SuperMarioBros-v3_data/img_data.csv')
        self.root_dir = 'SuperMarioBros-v3_data/images'

    def __len__(self):
        return len(self.images_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = []
        for i in range(num_frames-1, -1, -1):
            if idx - i < 0:
                j = 0
            else:
                j = i
            img_name = os.path.join(self.root_dir, self.images_csv.iloc[idx-j, 0])

            # todo: this should also be globally accessible so the trainer and tester can use the same things...
            image = Image.open(img_name)
            # image = image.crop((0, image.height / 2, image.width, image.height))
            # image.save('lol2.jpg')
            # https://www.geeksforgeeks.org/python-pil-image-resize-method/
            image = image.resize((image.width // 4, image.height // 4), 0)
            # todo: i'd like to downsample and having more of a "hard" color
            # basically, i want it black or white, and gray should be rounded...
            # image.save('lol1.jpg')
            # arr = arr.astype(dtype=np.dtype('f4'))
            tt = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor()])
            img = tt(image)
            imgs.append(img)
        # arr = np.array(image.convert('L'))
        # arr.resize((1, 210, 160)) # 1 channel, 210 height, 160 width
        # https://www.quora.com/Why-does-my-convolutional-neural-network-always-produce-the-same-outputs


        return torch.cat(imgs, 0)
        # return np.array(image.convert('L'))

adv_dataset = AdviceDataset()

# print(adv_dataset[0].shape)
height = adv_dataset[0].shape[1]
width = adv_dataset[0].shape[2]

# model = torch.nn.Sequential(
#     torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5),
#     torch.nn.ReLU()#,
#     # torch.nn.Linear(69, 3)
# )
#
# print(model(adv_dataset[0]).shape)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
torch.nn.init.xavier_uniform(conv1.weight)
lin1 = torch.nn.Linear((height-4) * (width-4) * 5, num_possible_actions)
torch.nn.init.xavier_uniform(lin1.weight)

model2 = torch.nn.Sequential(
    conv1,
    torch.nn.ReLU(),
    Flatten(),
    lin1, # use the formula for CNN shape!
    torch.nn.Softmax() #softmax or sigmoid???
)

# model3 = torch.nn.Sequential(
#     torch.nn.Conv3d(in_channels=1, out_channels=5, kernel_size=3),
#     torch.nn.ReLU(),
#     Flatten(),
#     torch.nn.Linear((height-2) * (width-2) * (num_frames-2) * 5, num_possible_actions), # use the formula for CNN shape!
#     torch.nn.Softmax() #softmax or sigmoid???
# )
#
# print(model2(adv_dataset[0]).shape)
y = adv_dataset.images_csv['action']
poss_actions = adv_dataset.images_csv['action'].unique()
df = pd.DataFrame()
for action in poss_actions:
    adv_dataset.images_csv[str(action)] = np.where(adv_dataset.images_csv['action']==action,
                                                   np.float32(1), np.float32(0))
    df[str(action)] = adv_dataset.images_csv[str(action)]

# weights = adv_dataset.images_csv['action'].value_counts()
# weights_balanced = (weights.sum() - weights) / weights.sum()
# weights_balanced = torch.tensor(np.array(weights_balanced)).float()
# weights_balanced = torch.tensor([0., 1., 1.]) # yikes...
dataloader = DataLoader(adv_dataset, batch_size=1, shuffle=True)
# give weight to each class in loss function for balance:

loss_fn = torch.nn.CrossEntropyLoss(
    # weight=weights_balanced
)
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-5)
# optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-5)

running_loss = 0

for epoch in range(5):
    print("epoch " + str(epoch))
    running_loss = 0
    misses = 0
    hits = 0
    # this is actually wrong, since we miss out on num_frames - 1 / num_frames training samples...
    for index, tensor_batch in enumerate(dataloader):
        # if len(tensor_batch) != num_frames:
        #     print("misses: " + str(misses) + ", hits: " + str(hits))
        #     # todo: these batches don't account for batching errors
        #     continue
        # print(model2(tensor_batch).shape)
        # print(index, tensor_batch.size())
        # Forward pass: compute predicted y by passing x to the model.
        # y_true = np.array(df[index*4:index*4+4])
        y_true = torch.tensor(np.array(df.iloc[index]))
        y_pred = model2(tensor_batch.reshape(1, 1, height, width))
        # if using 3d CNN, this should be 5D instead of 4D input

        # Compute and print loss.
        # todo: should i use argmax here? i dont think so...
        # https://github.com/pytorch/pytorch/issues/5554
        y_true_modded = torch.argmax(y_true).reshape((1,))

        if index == len(adv_dataset)-1:
            print(running_loss)
            print("misses: " + str(misses) + ", hits: " + str(hits))

        if y_true_modded[0] == 0:
            continue

        # y_true_modded = y_true_modded.double()
        # y_true_modded.requires_grad_(True)
        loss = loss_fn(y_pred, y_true_modded) # just compare the last output
        # print(loss.item())
        if torch.argmax(y_pred).item() != y_true_modded[0] and y_true_modded[0] != 0:
            misses += 1
        if torch.argmax(y_pred).item() == y_true_modded[0] and y_true_modded[0] != 0:
            hits += 1

        # if y_true_modded[0] != 0:
        #     print(y_pred, y_true)

        optimizer.zero_grad()
        loss.retain_grad()
        loss.backward()
        # print(loss.grad)
        optimizer.step()

        running_loss += loss.item() # this detaches the tensor???

torch.save(model2.state_dict(), "mario-test.model")