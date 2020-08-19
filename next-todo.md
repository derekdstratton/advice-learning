### GOALS
1. Get my API functioning at a basic level.
2. Try out things in my jupyter notebook to see what pure advice works best.
3. Design a script similar to `test_advice_model.py` that trains a RL model
to play (maybe called `train_deepq_model.py`)
4. Incorporate the advice_model to help train deepq faster, as an easy to test
script that takes in a trained advice model (probably a parameter in `train_deepq_model`)

-----

### TODO LIST
1. Train with test and validation set
2. Try different model architectures (more conv layers, dropout layers, etc)
3. Generalize to different games (will need to load parts based on games)
4. Make modular approach to preprocessing the training data. This would be pretty unique to each game
(such as removing deaths).
5. Other configurables: loss function, learning rate, random sampler methods for dataloader, different batch sizes,
optimizers

-----

### Future/More Advanced Problems to Solve:
it'd be fun to try things out with tas inputs

Does there need to be an intermediate step:
First, detect the objects and their locations. (unsupervised?)
http://e2crawfo.github.io/pdfs/spair_aaai_2019.pdf
Second, use this info with action labels.

It's even harder since the only relative locality that matters
is such that the ball is within this really far reach of the
paddle...

another thing to consider is that right now it really likes to do things that
are uncommon since i weighted all the training the same with the weighted
random sampler... Maybe I should try weighting things not all exactly the same,
but some sort of intermediary between totally imbalanced and completely balanced...

Small object detection:
https://medium.com/datadriveninvestor/small-objects-detection-problem-c5b430996162

Add interactivity back so it trains with inputs in the future

Consider periodically saving model state_dicts to see how different amounts of training affect performance