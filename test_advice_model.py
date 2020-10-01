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
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader

class EpisodeDataset(Dataset):
    def __init__(self, x, y):
        x = torch.stack(x)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], int(self.y[idx].item())

# todo: better style to make another method that takes in a model and an info object. more flexible

def test_model_on_game(model, training_info, num_epochs=100, visualize=True, output_directory=None, verbose=False):
    # get all of the training info
    model.cpu()

    if verbose:
        print("Visualizing" if visualize else "Not Visualizing")

    # load all the information from the model and the info
    adv_dataset = advice_dataset.AdviceDataset(training_info['dataset_path'], training_info['num_frames'],
                                               training_info['img_processor'])
    img_processor = getattr(image_processors, training_info['img_processor'])

    # create the game environment
    env = gym.make(training_info['game'])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # collect some data for analysis
    x_pos_arr = np.zeros(num_epochs)

    # the main loop for running the game
    for episode in range(0, num_epochs):
        # todo: consider if better as array or list?
        episode_history_x = []
        episode_history_y = []

        farthest = 0
        state = env.reset()
        # a sliding window list containing the frames up to the current one to analyze
        prev_frames = []
        image = Image.fromarray(state)
        img = img_processor(image)
        # this list should be populated with num_frames frames at all times
        for i in range(training_info['num_frames']):
            prev_frames.append(img)

        if verbose:
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

            # print(y_pred)
            # unweighted, deterministic sample best action (it can get stuck easily if its not perfect)
            # state2, reward, done, info = env.step(y_pred.detach().numpy().argmax())

            # if they don't sum to 1, maybe divide by the sum of all
            y_pred = y_pred / y_pred.sum()
            # weighted, random sampling to choose action
            weights = y_pred.detach().numpy().reshape(adv_dataset.num_possible_actions, )  # maybe this probability can
            # also be adjusted based on the reward?
            action = np.random.choice(np.arange(0, adv_dataset.num_possible_actions), p=weights)
            # TODO: add possible keyboard actions to train on

            state2, reward, done, info = env.step(action)

            episode_history_x.append(torch.cat(prev_frames))
            episode_history_y.append(action)

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
                if verbose:
                    print("Episode Reward: " + str(episode_reward))
                    print("Ending X Pos: " + str(farthest))
                    # todo: time is currently all wrong.
                    print("Ending Time: " + str(info['time']))
                x_pos_arr[episode] = farthest

                train = True
                if episode >= 1:
                    last_10_avg = np.mean(x_pos_arr[episode-1:episode])
                else:
                    last_10_avg = 1000
                print("Last 10 average: " + str(last_10_avg))
                # todo: right now just saayin if farthest > 1000, which is not robust but might show results now????
                if train and farthest > 1000:
                    loss_fn = torch.nn.BCELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    # assuming just 1 epoch of training? lower the learning rate a bit
                    dataset = EpisodeDataset(episode_history_x, episode_history_y)
                    weights = np.array([np.count_nonzero(dataset.y.numpy()==aa) for aa in range(0, 7)])
                    weights_balanced = 1. / weights
                    actions = dataset.y.numpy().astype(dtype=np.int32)
                    samples_weight = torch.tensor(np.array(weights_balanced)[actions])
                    # samples_weight = samples_weight.to(dev)
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    # return dataset, model # debugging
                    loader = DataLoader(dataset, sampler=sampler)

                    # todo: this data is still unbalanced in episode_history. and needs sampled.
                    print('training')
                    for epoch in range(0, 40):
                        print(epoch)
                        for x, y_action in loader:
                            # x.to(dev)
                            # y_chosen.to(dev)
                            y_model = model(x)
                            y_chosen = np.zeros(adv_dataset.num_possible_actions)
                            y_chosen[y_action] = 1
                            y_chosen = torch.Tensor(y_chosen)
                            loss = loss_fn(y_model, y_chosen)
                            # update the weights based on the loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    print('done training')
                break
    env.close()

    df = pd.DataFrame(x_pos_arr, columns=["x_pos"])

    # output a json of statistics to analyze (metrics), episode reward array, ending x pos array,
    if output_directory is not None:
        df.to_csv(output_directory + "/test-results.csv")

    return df, model


def test_model_on_game_from_file(model_input_directory, num_epochs=100, visualize=True, verbose=False):
    # get all of the training info
    with open(model_input_directory + "/info.json", "r") as fp:
        training_info = json.load(fp)

    # load all the information from the model and the info
    adv_dataset = advice_dataset.AdviceDataset(training_info['dataset_path'], training_info['num_frames'],
                                               training_info['img_processor'])
    model = models.AdviceModelGeneral(adv_dataset.img_height, adv_dataset.img_width, adv_dataset.num_possible_actions,
                                      training_info['num_frames'], training_info['num_layers'])

    # todo: this map device thing is for transferring models trained on a gpu to reloading on a cpu only.
    # prob should make it more dynamic
    model.load_state_dict(torch.load(model_input_directory + "/model.pt", map_location=torch.device('cpu')))
    return test_model_on_game(model, training_info, num_epochs, visualize,
                       output_directory=model_input_directory,
                       verbose=verbose)

if __name__ == "__main__":
    # in the future, parse command line args for convenience running files
    df, model = test_model_on_game_from_file("models/model9", num_epochs=200, visualize=False, verbose=True)