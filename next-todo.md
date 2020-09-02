### GOALS
1. Get my API functioning at a basic level.
2. Try out things in my jupyter notebook to see what pure advice works best.
3. Design a script similar to `test_advice_model.py` that trains a RL model
to play (maybe called `train_deepq_model.py`)
4. Incorporate the advice_model to help train deepq faster, as an easy to test
script that takes in a trained advice model (probably a parameter in `train_deepq_model`)

* What if we used a trained model (trained from deepq) as the base of the rl. and it just does
rl like normal, except using that as the base? im wondering if that would help at all? Like that's
the way it makes predictions?

-----

### TODO LIST
1. Train with test and validation set
    - Random split creates subsets, which are awkward to use
    with my custom dataset class
    - I could probably make some Dataset that takes in the
    original dataset and the indices to wrap it and obtain
    the correct values while still correctly exposing 
    variables, but it's bad design for sure.
    - when printing accuracy, i think these 2 values are useful. 1 would be absolutely if it got
    the prediction right or wrong (add 1 if wrong add 0 if right). Another, better way would be
    to take the probability given to the correct one, and do 1 - x to get the error between the
    expected and actual, and that gives more credit to being "on the right track", or possibly
    helping to not be as brutal with conflicting data
2. Try different model architectures (more conv layers, dropout layers, pooling, etc)
    - This is pretty good. It would be nice to make it more automatic as optional 
    parameters to pass to the functions. So diff architectures can easily be tested
    in a notebook.
    - 2 model functions: 1 for 2D conv, 1 for 3D conv (eventually later i can consider
    combining them, but for now i want to focus on 2D)
3. Generalize to different games (will need to load parts based on games)
4. Make modular approach to preprocessing the training data. This would be pretty unique to each game
(such as removing deaths).
5. Other configurables: loss function, learning rate, random sampler methods for dataloader, different batch sizes,
optimizers
6. Get GPUs working
    - they work, but theyre slow unless batches of >1 are used so we can actually use
    it lol
7. create a `requirements.txt` or something for easy installation: i just needed torch,
torchvision, and pandas for train. test needs gym, gym-super-mario-bros, idk about create
8. make jupyter notebooks also run remotely :D
9. To better my efficiency, instead of having a static number of epochs, dynamically choose the
number of epochs to keep going until you aren't learning any more

#### Convenience/Style
- make it so git lfs tracks SuperMario dataset (i messed it up)

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

try to investigate more about whats going on in the images from a computer vision 
perspective: filters, intermediate representations, etc.

another thing to consider is that right now it really likes to do things that
are uncommon since i weighted all the training the same with the weighted
random sampler... Maybe I should try weighting things not all exactly the same,
but some sort of intermediary between totally imbalanced and completely balanced...

Small object detection:
https://medium.com/datadriveninvestor/small-objects-detection-problem-c5b430996162

Add interactivity back so it trains with inputs in the future

Consider periodically saving model state_dicts to see how different amounts of training affect performance