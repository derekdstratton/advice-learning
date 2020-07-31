# https://pytorch.org/tutorials/beginner/saving_loading_models.html
import time

import gym
import torch
import torchvision
from PIL import Image
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from torch import nn
import numpy as np
import models
import image_processors
import advice_dataset

# todo: make command line arguments
DATASET_PATH = "SuperMarioBros-v3_data"
OUTPUT_PATH = "mario-test.model"
MODEL = "AdviceModel"
IMG_PROCESSOR = "downsample"
NUM_EPOCHS = 100
NUM_FRAMES = 1

adv_dataset = advice_dataset.AdviceDataset(DATASET_PATH, NUM_FRAMES)

model = getattr(models, MODEL)(adv_dataset.img_height, adv_dataset.img_width, adv_dataset.num_possible_actions, NUM_FRAMES)

game = 'SuperMarioBros-v3'
env = gym.make(game)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
model.load_state_dict(torch.load(OUTPUT_PATH))

for episode in range(0, NUM_EPOCHS):
    state = env.reset()

    prev_frames = []
    image = Image.fromarray(state)
    img = getattr(image_processors, IMG_PROCESSOR)(image)
    for i in range(NUM_FRAMES):
        prev_frames.append(img)

    print("Episode " + str(episode))
    episode_reward = 0
    step = 0
    while True:
        try:
            env.render()
            image = Image.fromarray(state)
            img = getattr(image_processors, IMG_PROCESSOR)(image)


            prev_frames.append(img)
            prev_frames.pop(0)
            action = model.forward(torch.cat(prev_frames))
            # print(action.detach().numpy())

            # unweighted, deterministic sample best action
            # state2, reward, done, info = env.step(action.detach().numpy().argmax())

            # weighted, random sampling to choose action
            weights = action.detach().numpy().reshape(adv_dataset.num_possible_actions,)
            action = np.random.choice(np.arange(0, adv_dataset.num_possible_actions), p=weights)
            print(action)
            state2, reward, done, info = env.step(action)

            step += 1
            state = state2

            episode_reward += reward

            time.sleep(0.008)
            if done or info['flag_get']:
                break
        except KeyboardInterrupt:
            print('dont key interrupr bro')
env.close()