import gym
# import keyboard
import time
# make a mapping of keys to actions (json?)

env = gym.make('Breakout-v0')
env.reset()

for i in range(0, 10):
    # convert rgb to grayscale
    state = env.reset()
    print("Episode " + str(i))

    while True:
        env.render()
        action = 1
        if keyboard.is_pressed('left'):
            action = 0
        elif keyboard.is_pressed('right'):
            action = 2
        state2, reward, done, info = env.step(action)
        # print(reward)
        time.sleep(0.04)
        if done:
            break
env.close()
