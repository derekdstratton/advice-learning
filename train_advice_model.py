from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import models
import advice_dataset
import os
import json

# in the future, create environment variables that represent directories
# in the future use pathlib instead of os ideally
MAIN_OUTPUT_DIRECTORY = "models"

# gpus! it's actually worse unless i start training in batches lol
print(gpu_avail := torch.cuda.is_available())

### WARNING: The testability of a model is wrecked if it's using cuda vs if it's using a cpu. just be careful not to
### remotely train on gpu then test locally on cpu, it will not work.
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO: make a configurable parameter for the option to output to a file or not
def train_model_on_dataset(model_name, game_name, dataset_path, img_processor, num_frames, num_epochs):
    # handle directories for output
    if not os.path.exists(MAIN_OUTPUT_DIRECTORY):
        os.makedirs(MAIN_OUTPUT_DIRECTORY)
    output_directory = MAIN_OUTPUT_DIRECTORY + "/" + game_name + "_" + model_name
    output_directory_to_check = output_directory
    index = 0
    while os.path.isdir(output_directory_to_check):
        output_directory_to_check = output_directory + "_" + str(index)
        index += 1
    os.mkdir(output_directory_to_check)
    output_directory = output_directory_to_check
    output_model_path = output_directory + "/model.pt"
    output_info_path = output_directory + "/info.json"

    # info is for the json object that stores info about the training
    info = {
        "num_frames": num_frames,
        "num_epochs": num_epochs,
        "game": game_name,
        "dataset_path": dataset_path,
        "model": model_name,
        "img_processor": img_processor
    }

    # set up the dataset
    adv_dataset = advice_dataset.AdviceDataset(dataset_path, num_frames, img_processor)

    # set up the model
    model = getattr(models, model_name)(adv_dataset.img_height, adv_dataset.img_width,
                                        adv_dataset.num_possible_actions, num_frames)
    model = model.to(dev)

    # sample the imbalanced data so that everything is weighted evenly ( 1 / weights )
    weights = adv_dataset.y.numpy().sum(axis=0)
    weights_balanced = 1. / weights
    actions = adv_dataset.images_csv['action']
    samples_weight = torch.tensor(np.array(weights_balanced)[actions])
    samples_weight = samples_weight.to(dev)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # create dataloader, loss function, and optimizer
    dataloader = DataLoader(adv_dataset, batch_size=1, sampler=sampler)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # main training loop
    for epoch in range(num_epochs):
        print("epoch " + str(epoch))
        running_loss = 0
        misses = 0
        hits = 0
        for index, data in enumerate(dataloader):
            x, y_true = data
            x = x.to(dev)
            y_true = y_true.to(dev)

            # get the model's current prediction
            y_pred = model(x)

            # compute the loss with CrossEntropyLoss, which takes the argmax of the true label
            loss = loss_fn(y_pred, torch.argmax(y_true).reshape((1,)))

            # update the weights based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # debugging
            # print(y_pred)

            # accumulate some data
            if torch.argmax(y_pred).item() == torch.argmax(y_true).item():
                hits += 1
            else:
                misses += 1
            running_loss += loss.item()

            # at the end of the epoch, print some stats
            if index == len(adv_dataset) - 1:
                print("Total Loss: " + str(running_loss))
                # print("Misses: " + str(misses) + ", Hits: " + str(hits))
                print("Accuracy: " + str(hits/len(adv_dataset)))

    # output info
    torch.save(model.state_dict(), output_model_path)
    with open(output_info_path, "w") as fp:
        json.dump(info, fp)
    print("Files output to: " + output_directory)
    # todo: consider outputting a json of statistics to analyze (metrics)
    return model, info

if __name__ == "__main__":
    train_model_on_dataset(
        model_name="AdviceModel",
        game_name="SuperMarioBros-v3",
        dataset_path="SuperMarioBros-v3_data",
        img_processor="downsample",
        num_frames=1,
        num_epochs=20
    )