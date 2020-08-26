

consider using parsec

running on windows wwith wsl (and pycharm):
https://dibranmulder.github.io/2019/09/06/Running-an-OpenAI-Gym-on-Windows-with-WSL/
DISPLAY=:0 important environment variable

Lab Machine: `ssh dstratton@134.197.95.144`

`export DISPLAY={my windows IP}:0`

can't test remotely with visuals in PyCharm since there's no x11 forwarding

running keyboard library as root (required for linux):

remotely connecting to another computer:
use private ip from hostname -I to connect. install openssh-server on it and follow a guide. ezpz

pycharm doesnt support x11, but you can do it with putty and it works. don't
be root, instead do sudo venv/bin/python3 run.py

Convenience for running as root:
./python_run.sh create_advice_dataset.py 

https://esmithy.net/2015/05/05/rundebug-as-root-in-pycharm/

Remote file transfer options:

https://intellij-support.jetbrains.com/hc/en-us/community/posts/207001235-Disable-automatic-upload

https://www.linuxjournal.com/content/share-keyboardmouse-between-multiple-computers-x2x

https://intellij-support.jetbrains.com/hc/en-us/community/posts/360003435280-X11-Forwarding-in-PyCharm\

some of these problems can be reasonably circumvented by connecting
a vm to pycharm instead of dual booting.