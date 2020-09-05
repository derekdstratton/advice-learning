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
import pandas as pd

# todo: better style to make another method that takes in a model and an info object. more flexible

def test_model_on_game(model, training_info, num_epochs=100, visualize=True, output_directory=None):
    # get all of the training info

    print(visualize)

    # load all the information from the model and the info
    adv_dataset = advice_dataset.AdviceDataset(training_info['dataset_path'], training_info['num_frames'],
                                               training_info['img_processor'])
    img_processor = getattr(image_processors, training_info['img_processor'])

    # create the game environment
    env = gym.make(training_info['game'])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # collect some data for analysis
    x_pos_arr = np.zeros(num_epochs)

    farthest = 0

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

            print(y_pred)
            # unweighted, deterministic sample best action (it can get stuck easily if its not perfect)
            # state2, reward, done, info = env.step(y_pred.detach().numpy().argmax())

            # if they don't sum to 1, maybe divide by the sum of all
            y_pred = y_pred / y_pred.sum()

            # weighted, random sampling to choose action
            weights = y_pred.detach().numpy().reshape(adv_dataset.num_possible_actions, )  # maybe this probability can
            # also be adjusted based on the reward?
            action = np.random.choice(np.arange(0, adv_dataset.num_possible_actions), p=weights)

            state2, reward, done, info = env.step(action)

            # debugging
            # print(y_pred.detach().numpy()) # the array of predicted actions at a given state
            # print(action)

            # save various statistics
            step += 1
            state = state2
            episode_reward += reward

            # sleep to make viewing easier if visualizing
            # if visualize:
            #     time.sleep(0.008)

            # use this cause ending x pos is always "40" if lives are < 2, since you reset
            farthest = max(farthest, info['x_pos'])

            # check for end of episode (for mario, it's either when you win or die)
            if done or info['flag_get'] or info['time'] <= 1 or info['life'] < 2:
                # this is the exit point.
                print("Episode Reward: " + str(episode_reward))
                print("Ending X Pos: " + str(farthest))
                print("Ending Time: " + str(info['time']))
                x_pos_arr[episode] = farthest
                break
    env.close()

    df = pd.DataFrame(x_pos_arr, columns=["x_pos"])
    print(df)

    # todo: output a json of statistics to analyze (metrics), episode reward array, ending x pos array,
    if output_directory is not None:
        df.to_csv(output_directory + "/output.csv")


def test_model_on_game_from_file(model_input_directory, num_epochs=100, visualize=True):
    # get all of the training info
    with open(model_input_directory + "/info.json", "r") as fp:
        training_info = json.load(fp)

    # load all the information from the model and the info
    adv_dataset = advice_dataset.AdviceDataset(training_info['dataset_path'], training_info['num_frames'],
                                               training_info['img_processor'])
    model = getattr(models, training_info['model'])(adv_dataset.img_height, adv_dataset.img_width,
                                                    adv_dataset.num_possible_actions, training_info['num_frames'])

    # todo: this map device thing is for transferring models trained on a gpu to reloading on a cpu only.
    # prob should make it more dynamic
    model.load_state_dict(torch.load(model_input_directory + "/model.pt", map_location=torch.device('cpu')))
    test_model_on_game(model, training_info, num_epochs, visualize, output_directory=model_input_directory)

if __name__ == "__main__":
    # in the future, parse command line args for convenience running files
    test_model_on_game_from_file("models/SuperMarioBros-v3_AdviceModel2Layer_6", num_epochs=1, visualize=True)