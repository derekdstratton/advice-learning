import os

import gym
# import keyboard
import time
# make a mapping of keys to actions (json?)
import keyboard as keyboard
from PIL import Image
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

game = 'SuperMarioBros-v3'
env = gym.make(game)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env.reset()
frame_id = 0
path = game + "_data" # make this configurable, probably set to the name of the game! Command Line option ideal

try:
    os.mkdir(path)
    os.mkdir(path + "/images")
except OSError:
    # todo: if file exists, print specific error for that
    print("RIPPERONI")

img_data_file_path = path + "/img_data.csv"
img_data = open(img_data_file_path, "w+")
img_data.write("img_file_name,action,episode,step,reward" + "\n")

episode_data_file_path = path + "/episode_data.csv"
episode_data = open(episode_data_file_path, "w+")
episode_data.write("episode,episode_reward"+ "\n")

for episode in range(0, 100):
    # convert rgb to grayscale
    state = env.reset()
    print("Episode " + str(episode))
    episode_reward = 0
    step = 0
    while True:
        try:
            env.render()
            action = 0
            if keyboard.is_pressed('left'):
                action = 6
            elif keyboard.is_pressed('right') and keyboard.is_pressed('z') and keyboard.is_pressed('x'):
                action = 4
            elif keyboard.is_pressed('right') and keyboard.is_pressed('z'):
                action = 2
            elif keyboard.is_pressed('right') and keyboard.is_pressed('x'):
                action = 3
            elif keyboard.is_pressed('z'):
                action = 5
            elif keyboard.is_pressed('right'):
                action = 1
            state2, reward, done, info = env.step(action)
            img = Image.fromarray(state2)
            img.save(path + "/images/" + str(frame_id) + ".jpg") # consider more idiomatic file path construction?

            img_data.write(str(frame_id) + ".jpg" + ", " + str(action) + ", " + str(episode) + ", " + str(step) + ", " + str(reward)+ "\n")

            frame_id += 1
            step += 1

            episode_reward += reward

            # print(state2)
            # print(reward)

            time.sleep(0.008)
            if done or info['flag_get']:
                break
            if keyboard.is_pressed('q'):
                episode_data.close()
                img_data.close()
                env.close()
                exit(1)
        except KeyboardInterrupt:
            print('dont key interrupr bro')
    episode_data.write(str(episode) + ", " + str(episode_reward)+ "\n")
env.close()
