

consider using parsec

running on windows wwith wsl (and pycharm):
https://dibranmulder.github.io/2019/09/06/Running-an-OpenAI-Gym-on-Windows-with-WSL/
DISPLAY=:0 important environment variable

Lab Machine: `ssh dstratton@134.197.95.144`

To make sure it doesn't fail, try using `screen`

I could also use `scp` to bring trained models back to this to test visually. Or I can just test
them remotely, either works.

copy a directory: `pscp -r -P 22 dstratton@134.197.95.144:/home/dstratton/PycharmProjects/advice-learning-remote/models/SuperMarioBros-v3_AdviceModel_21 .`
^ optionally run with `-pw password` for easier life

`export DISPLAY={my windows IP}:0`

local ip: `172.27.42.239`

can't test remotely with visuals in PyCharm since there's no x11 forwarding

running keyboard library as root (required for linux):

remotely connecting to another computer:
use private ip from hostname -I to connect. install openssh-server on it and follow a guide. ezpz

pycharm doesnt support x11, but you can do it with putty and it works. don't
be root, instead do sudo venv/bin/python3 run.py

In Putty, set X11 forwarding to true, and in the box put `{local ip}:0` as the destination and it'll 
work. Also make sure to turn xming on in windows. it sucks if i have bad internet lol

Convenience for running as root:
./python_run.sh create_advice_dataset.py 

https://esmithy.net/2015/05/05/rundebug-as-root-in-pycharm/

Remote file transfer options:

https://intellij-support.jetbrains.com/hc/en-us/community/posts/207001235-Disable-automatic-upload

https://www.linuxjournal.com/content/share-keyboardmouse-between-multiple-computers-x2x

https://intellij-support.jetbrains.com/hc/en-us/community/posts/360003435280-X11-Forwarding-in-PyCharm\

some of these problems can be reasonably circumvented by connecting
a vm to pycharm instead of dual booting.