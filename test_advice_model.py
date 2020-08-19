import json
import time
import gym
import torch
from PIL import Image
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import models
import image_processors
import advice_dataset

# todo: better style to make another method that takes in a model and an info object. more flexible

def test_model_on_game(model, training_info, num_epochs=100, visualize=True):
    # get all of the training info

    # load all the information from the model and the info
    adv_dataset = advice_dataset.AdviceDataset(training_info['dataset_path'], training_info['num_frames'],
                                               training_info['img_processor'])
    img_processor = getattr(image_processors, training_info['img_processor'])

    # create the game environment
    env = gym.make(training_info['game'])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # the main loop for running the game
    for episode in range(0, num_epochs):
        state = env.reset()
        # a sliding window list containing the frames up to the current one to analyze
        prev_frames = []
        image = Image.fromarray(state)
        img = img_processor(image)
        # this list should be populated with num_frames frames at all times
        for i in range(training_info['num_frames']):
            prev_frames.append(img)

        print("Episode " + str(episode))
        episode_reward = 0
        step = 0
        while True:
            if visualize:
                env.render()
            image = Image.fromarray(state)
            img = img_processor(image)

            # add the new frame on and remove the old frame from the sliding window list
            prev_frames.append(img)
            prev_frames.pop(0)
            y_pred = model.forward(torch.cat(prev_frames))

            # unweighted, deterministic sample best action (it can get stuck easily if its not perfect)
            # state2, reward, done, info = env.step(y_pred.detach().numpy().argmax())

            # weighted, random sampling to choose action
            weights = y_pred.detach().numpy().reshape(adv_dataset.num_possible_actions, )  # maybe this probability can
            # also be adjusted based on the reward?
            action = np.random.choice(np.arange(0, adv_dataset.num_possible_actions), p=weights)

            state2, reward, done, training_info = env.step(action)

            # debugging
            # print(y_pred.detach().numpy()) # the array of predicted actions at a given state
            # print(action)

            # save various statistics
            step += 1
            state = state2
            episode_reward += reward

            # sleep to make viewing easier if visualizing
            if visualize:
                time.sleep(0.008)

            # check for end of episode (for mario, it's either when you win or die)
            if done or training_info['flag_get'] or training_info['time'] <= 1 or training_info['life'] < 2:
                # this is the exit point.
                print("Episode Reward: " + str(episode_reward))
                print("Ending X Pos: " + str(training_info['x_pos']))
                print("Ending Time: " + str(training_info['time']))
                break
        env.close()
        # todo: output a json of statistics to analyze (metrics), episode reward array, ending x pos array,

def test_model_on_game_from_file(model_input_directory, num_epochs=100, visualize=True):
    # get all of the training info
    with open(model_input_directory + "/info.json", "r") as fp:
        training_info = json.load(fp)

    # load all the information from the model and the info
    adv_dataset = advice_dataset.AdviceDataset(training_info['dataset_path'], training_info['num_frames'],
                                               training_info['img_processor'])
    model = getattr(models, training_info['model'])(adv_dataset.img_height, adv_dataset.img_width,
                                                    adv_dataset.num_possible_actions, training_info['num_frames'])
    model.load_state_dict(torch.load(model_input_directory + "/model.pt"))
    test_model_on_game(model, training_info, num_epochs, visualize)

if __name__ == "__main__":
    # in the future, parse command line args for convenience running files
    test_model_on_game_from_file("models/SuperMarioBros-v3_AdviceModel_5", num_epochs=1, visualize=True)