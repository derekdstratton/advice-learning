from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, SubsetRandomSampler
import torch
import numpy as np
import pandas as pd
import models
import advice_dataset
import os
import json

# i think this works, like 99% sure. really hope it does
class SubsetWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, weights, indices):
        super().__init__(weights, len(indices))
        self.indices = indices

    def __iter__(self):
        weights_perm = torch.zeros(len(self.indices))
        it = 0
        for i in self.indices:
            weights_perm[it] = self.weights[i]
            it += 1
        mn = torch.multinomial(weights_perm, self.num_samples, self.replacement).tolist()
        return (self.indices[k] for k in mn)

    def __len__(self):
        return len(self.indices)

def train_model_on_dataset(model_name, game_name, dataset_path, img_processor, num_frames, num_epochs,
                           num_layers, output_directory=None, verbose=False):
    if output_directory is not None:
        os.mkdir(output_directory)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if verbose:
        print("Using device: " + str(dev))

    # info is for the json object that stores info about the training
    info = {
        "num_frames": num_frames,
        "num_epochs": num_epochs,
        "num_layers": num_layers,
        "game": game_name,
        "dataset_path": dataset_path,
        "model": model_name,
        "img_processor": img_processor
    }

    # set up the dataset
    adv_dataset = advice_dataset.AdviceDataset(dataset_path, num_frames, img_processor)

    # set up the model
    model = models.AdviceModelGeneral(adv_dataset.img_height, adv_dataset.img_width,
                                      adv_dataset.num_possible_actions, num_frames, num_layers=num_layers)
    # todo: now that i've generalized this to 1 model, i should just call it by name.
    # model = getattr(models, model_name)(adv_dataset.img_height, adv_dataset.img_width,
    #                                     adv_dataset.num_possible_actions, num_frames)
    model = model.to(dev)

    # sample the imbalanced data so that everything is weighted evenly ( 1 / weights )
    weights = adv_dataset.y.numpy().sum(axis=0)
    weights_balanced = 1. / weights
    actions = adv_dataset.images_csv['action']
    samples_weight = torch.tensor(np.array(weights_balanced)[actions])
    samples_weight = samples_weight.to(dev)

    ### Random split into training and test set, with dataloaders for each
    train_len = int(0.8 * len(adv_dataset))
    train, val = random_split(adv_dataset, [train_len, len(adv_dataset)-train_len])
    train_sampler = SubsetWeightedRandomSampler(samples_weight, train.indices)
    validation_sampler = SubsetRandomSampler(val.indices)
    train_loader = DataLoader(adv_dataset, sampler=train_sampler)
    val_loader = DataLoader(adv_dataset, sampler=validation_sampler)

    # create loss function and optimizer
    # use bce or softmax?
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    training_accs = np.zeros(num_epochs)
    val_accs = np.zeros(num_epochs)

    # main training loop
    for epoch in range(num_epochs):
        if verbose:
            print("epoch " + str(epoch))
        running_loss = 0
        training_hits = 0
        val_hits = 0
        running_val_loss = 0
        # training
        for index, data in enumerate(train_loader):
            x, y_true = data
            x = x.to(dev)
            y_true = y_true.to(dev)

            # get the model's current prediction
            y_pred = model(x)

            # compute the loss with CrossEntropyLoss, which takes the argmax of the true label
            # loss = loss_fn(y_pred, torch.argmax(y_true).reshape((1,)))
            # BCELoss
            loss = loss_fn(y_pred.double(), y_true.double())

            # update the weights based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # debugging
            # print(y_pred)

            # accumulate some data
            if torch.argmax(y_pred).item() == torch.argmax(y_true).item():
                training_hits += 1
            running_loss += loss.item()

            # at the end of the epoch, print some stats
            if index == len(train_sampler.indices) - 1:
                if verbose:
                    print("Training Loss: " + str(running_loss))
                    print("Train Accuracy: " + str(training_hits/len(train_sampler.indices)))
                training_losses[epoch] = running_loss
                training_accs[epoch] = training_hits/len(train_sampler.indices)
        # validation
        for index, data in enumerate(val_loader):
            x, y_true = data
            x = x.to(dev)
            y_true = y_true.to(dev)

            # get the model's current prediction
            y_pred = model(x)
            running_val_loss += loss_fn(y_pred.double(), y_true.double()).item()
            # accumulate some data
            if torch.argmax(y_pred).item() == torch.argmax(y_true).item():
                val_hits += 1
            if index == len(validation_sampler.indices) - 1:
                if verbose:
                    print("Val Loss: " + str(running_val_loss))
                    print("Val Accuracy: " + str(val_hits/len(validation_sampler.indices)))
                val_losses[epoch] = running_val_loss
                val_accs[epoch] = val_hits/len(validation_sampler.indices)

    # concatenate all the training and validation metrics into a dataframe
    df = pd.DataFrame({"training_losses": training_losses,
                         "training_acc": training_accs,
                         "val_losses": val_losses,
                         "val_acc": val_accs})
    # output to file
    if output_directory is not None:
        output_model_path = output_directory + "/model.pt"
        output_info_path = output_directory + "/info.json"
        output_training_path = output_directory + "/training-metrics.csv"
        df.to_csv(output_training_path)

        torch.save(model.state_dict(), output_model_path)
        with open(output_info_path, "w") as fp:
            json.dump(info, fp)
        if verbose:
            print("Files output to: " + output_directory)
    return model, info, df

if __name__ == "__main__":
    model,info,df=train_model_on_dataset(
        model_name="AdviceModelGeneral",
        game_name="SuperMarioBros-v3",
        dataset_path="SuperMarioBros-v3_data",
        img_processor="downsample",
        num_frames=1,
        num_epochs=100,
        num_layers=3,
        output_directory="models/model8",
        verbose=True
    )