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

num_possible_actions = 7
height = 60
width = 64
# todo: these values should all be stored in some sort of external model class to be used by trainer and tester

conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
torch.nn.init.xavier_uniform(conv1.weight)
lin1 = torch.nn.Linear((height-4) * (width-4) * 5, num_possible_actions)
torch.nn.init.xavier_uniform(lin1.weight)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

model2 = torch.nn.Sequential(
    conv1,
    torch.nn.ReLU(),
    Flatten(),
    lin1, # use the formula for CNN shape!
    torch.nn.Softmax() #softmax or sigmoid???
)

game = 'SuperMarioBros-v3'
env = gym.make(game)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
model2.load_state_dict(torch.load("mario-test.model"))

for episode in range(0, 100):
    # convert rgb to grayscale
    state = env.reset()
    print("Episode " + str(episode))
    episode_reward = 0
    step = 0
    while True:
        try:
            env.render()

            # from train...
            image = Image.fromarray(state)
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


            action = model2.forward(img.reshape(1, 1, height, width))
            print(action.detach().numpy())
            state2, reward, done, info = env.step(action.detach().numpy().argmax())
            step += 1
            state = state2

            episode_reward += reward


            time.sleep(0.008)
            if done or info['flag_get']:
                break
        except KeyboardInterrupt:
            print('dont key interrupr bro')
env.close()