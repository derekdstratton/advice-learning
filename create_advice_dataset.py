import os

import gym
# import keyboard
import time
# make a mapping of keys to actions (json?)
import keyboard as keyboard
from PIL import Image

env = gym.make('Breakout-v0')
env.reset()
frame_id = 0
path = "data" # make this configurable, probably set to the name of the game! Command Line option ideal

try:
    os.mkdir(path)
    os.mkdir(path + "/images")
except OSError:
    # todo: if file exists, print specific error for that
    print("RIPPERONI")

img_data_file_path = path + "/img_data.csv"
img_data = open(img_data_file_path, "w+")
img_data.write("img_file_name, action, episode, step, reward" + "\n")

episode_data_file_path = path + "/episode_data.csv"
episode_data = open(episode_data_file_path, "w+")
episode_data.write("episode, episode_reward"+ "\n")

episode_reward = 0
for episode in range(0, 3):
    # convert rgb to grayscale
    state = env.reset()
    print("Episode " + str(episode))

    step = 0
    while True:
        env.render()
        action = 1
        if keyboard.is_pressed('left'):
            action = 3
        elif keyboard.is_pressed('right'):
            action = 2
        state2, reward, done, info = env.step(action)
        img = Image.fromarray(state2)
        img.save(path + "/images/" + str(frame_id) + ".jpg") # consider more idiomatic file path construction?

        img_data.write(str(frame_id) + ".jpg" + ", " + str(action) + ", " + str(episode) + ", " + str(step) + ", " + str(reward)+ "\n")

        frame_id += 1
        step += 1

        episode_reward += reward

        # print(state2)
        # print(reward)

        time.sleep(0.04)
        if done:
            break
    episode_data.write(str(episode) + ", " + str(episode_reward)+ "\n")
env.close()
