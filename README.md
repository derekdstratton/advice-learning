# Advice Learning

### Description

There are 3 main modules to run, usually in this order:

- `create_advice_dataset.py`: creates a dataset for a game based
on a human player playing the game and saving the inputs (advice)
- `train_advice_model.py`: trains a model to learn how to play a
game based on a dataset
- `test_advice_model.py`: runs a trained advice model on the game
to visually see its performance and also to see the reward it earns

### Requirements

- Everything has been run and tested in a Linux environment
with Python 3.8
- `create_advice_dataset.py`: 
    - Some dependencies
    - You need access to a graphical interface
    - You must have access to the keyboard of the machine 
    - You must be root (to run the keyboard library)
    this is being run on
- `train_advice_model.py`:
    - Some dependencies
- `test_advice_model.py`:
    - Some dependencies
    - You need access to a graphical interface

### Usage

`python`

csv file format:
img_file_name, action, episode, step, reward

reqs
gym[atari] for breakout
keyboard
gym


create_advice_dataset should be a class that can override methods particular to certain use
cases (for example, override the ending condition for things specific to mario like 'flag_get')
