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

adv_dataset = advice_dataset.AdviceDataset(DATASET_PATH)

model2 = getattr(models, "AdviceModel")(adv_dataset.img_height, adv_dataset.img_width, adv_dataset.num_possible_actions)

game = 'SuperMarioBros-v3'
env = gym.make(game)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
model2.load_state_dict(torch.load(OUTPUT_PATH))

for episode in range(0, 100):
    state = env.reset()
    print("Episode " + str(episode))
    episode_reward = 0
    step = 0
    while True:
        try:
            env.render()

            image = Image.fromarray(state)
            img = image_processors.downsample(image)

            action = model2.forward(img.reshape(1, 1, adv_dataset.img_height, adv_dataset.img_width))
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