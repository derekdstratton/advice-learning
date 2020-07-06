csv file format:
img_file_name, action, episode, step, reward

reqs
gym[atari] for breakout
keyboard
gym

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