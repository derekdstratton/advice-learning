I FINALLY DID IT OMG I THOUGHT I TRIED IT:

First set up a local port forward: (todo: put my password in the command)
`ssh -L 6000:134.197.95.144:22 derekstratton@ubuntu.cse.unr.edu`
I99J&^j91q8fbfnQJld$


Then just connect!
`ssh -p 6000 dstratton@localhost`

If you get `key_identification_error`, get into my
machine and `sudo reboot` it. It means a connection was
poorly closed maybe. Or it means you closed the terminal with the
local port open. so do that again? idk honestly.

I think it's okay if the client computer's display tunrs
off, but it must not go to sleep (change settings).

Bringing files back: (todo: put my password in the command)
`scp -P 6000 -r dstratton@localhost:PycharmProjects/advice-learning-remote/models/SuperMarioBros-v3_AdviceModel_21 models`
consider using parsec

There seems to be some interference with port forwarding and X forwarding????
If local testing with X doesn't work, clsoe ports?

running on windows wwith wsl (and pycharm):
https://dibranmulder.github.io/2019/09/06/Running-an-OpenAI-Gym-on-Windows-with-WSL/
DISPLAY=:0 important environment variable

Lab Machine: `ssh dstratton@134.197.95.144`

https://medium.com/@kennch/configure-a-transparent-multi-hop-ssh-connection-fe63437f5a33::wq
`ssh -A -t derekstratton@ubuntu.cse.unr.edu ssh -A dstratton@134.197.95.144`

^ note, this only works on eduroam, but to get it to work on ethernet i might need to tunnel to 
the network

ssh derekstratton@ubuntu.cse.unr.edu

ssh -L 0.0.0.0:2222:localhost:22 derekstratton@ubuntu.cse.unr.edu
ssh -p 2222 derekstratton@localhost
Actually, I think I can just set up a local port that can be my lab computer...

ssh dynamic port forwarding?

I can tunnel to my lab machine through there, from anywhere!

try configuring pycharm to use putty to tunnel through with some private key file:
https://www.jetbrains.com/help/idea/configuring-ssh-and-ssl.html#ssh

hippity hoppity with putty or ssh:
https://serverfault.com/questions/340865/ssh-tunnel-over-multi-hops-using-putty
https://superuser.com/questions/96489/an-ssh-tunnel-via-multiple-hops

openssh: maybe in the future for windows but prob sticking with putty
https://www.maketecheasier.com/use-windows10-openssh-client/

To make sure it doesn't fail, try using `screen`

I could also use `scp` to bring trained models back to this to test visually. Or I can just test
them remotely, either works.

copy a directory: `pscp -r -P 22 dstratton@134.197.95.144:/home/dstratton/PycharmProjects/advice-learning-remote/models/SuperMarioBros-v3_AdviceModel_21 .`
^ optionally run with `-pw password` for easier life

`export DISPLAY={my windows IP}:0`

`export DISPLAY=:0` <-- this command saves lives. for wsl

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