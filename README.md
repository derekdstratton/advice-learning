csv file format:
img_file_name, action, episode, step, reward

reqs
gym[atari] for breakout
keyboard
gym

consider using parsec

it'd be fun to try things out with tas inputs

create_advice_dataset should be a class that can override methods particular to certain use
cases (for example, override the ending condition for things specific to mario like 'flag_get')

running on windows wwith wsl (and pycharm):
https://dibranmulder.github.io/2019/09/06/Running-an-OpenAI-Gym-on-Windows-with-WSL/
DISPLAY=:0 important environment variable

running keyboard library as root (required for linux):

remotely connecting to another computer:
use private ip from hostname -I to connect. install openssh-server on it and follow a guide. ezpz

pycharm doesnt support x11, but you can do it with putty and it works. don't
be root, instead do sudo venv/bin/python3 run.py

(in putty, don't put an input for x11 forward location, and make sure xming is open)

probably just don't allow giving remote commands... for now...
unfortunately, keyboard always runs using your own os. a way around it
would be to make a script that runs locally and checks for key presses,
and then remotely starts a process
https://stackoverflow.com/questions/946946/how-to-execute-a-process-remotely-using-python

some of these problems can be reasonably circumvented by connecting
a vm to pycharm instead of dual booting.

https://www.linuxjournal.com/content/share-keyboardmouse-between-multiple-computers-x2x

https://intellij-support.jetbrains.com/hc/en-us/community/posts/360003435280-X11-Forwarding-in-PyCharm\

Convenience for running as root:
./python_run.sh create_advice_dataset.py 

i think when i create the original dataset, i shouldn't modify the original images.

and then when training/analyzing, you can modify them.

i also think that there should be scripts that can be used to "clean up" the bad data. like if you miss,
specific to breakout, you can see the score down. you can probably remove that bit of data manually
from training by detecting image changes from the top. almost like creating an additional label after the
training phase.

https://esmithy.net/2015/05/05/rundebug-as-root-in-pycharm/

image detection on this problem:

I feel like the differences in the images are really small,
which makes it not really learn... Tiny discrepancies mean that it's
always losing, and it's not really getting any closer.

Is ~500 left or right input images too few? Does this problem
need more data? More layers in the CNN? Better layers?

Is a 3D CNN too complex to quickly find the small ball and
paddle both moving at the same time?

Does there need to be an intermediate step:
First, detect the objects and their locations. (unsupervised?)
http://e2crawfo.github.io/pdfs/spair_aaai_2019.pdf
Second, use this info with action labels.

It's even harder since the only relative locality that matters
is such that the ball is within this really far reach of the
paddle... 

https://medium.com/datadriveninvestor/small-objects-detection-problem-c5b430996162

Remote file transfer options:

https://intellij-support.jetbrains.com/hc/en-us/community/posts/207001235-Disable-automatic-upload

