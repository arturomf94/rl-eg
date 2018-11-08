### Code from http://kvfrans.com/simple-algoritms-for-solving-cartpole/

import gym
import numpy as np


# Define environment:

env = gym.make('CartPole-v0')
# env.reset()

# Function to run episode:

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

# Hill climb!

noise_scaling = 0.1
parameters = np.random.rand(4) * 2 - 1
bestreward = 0
for _ in range(10000):
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
    reward = 0
    run = run_episode(env, newparams)
    if reward > bestreward:
        bestreward = reward
        parameters = newparams
        if reward == 200:
            break


# Run two hundred steps with best parameters

observation = env.reset()
for t in range(200):
    env.render()
    action = 0 if np.matmul(parameters, observation) < 0 else 1
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
