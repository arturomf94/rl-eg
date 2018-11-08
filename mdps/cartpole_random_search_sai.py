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

# Random search the space of parameters.

bestparams = None
bestreward = 0
for _ in range(10000):
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env, parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break


# Run two hundred steps with best parameters

observation = env.reset()
for t in range(200):
    env.render()
    action = 0 if np.matmul(bestparams, observation) < 0 else 1
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
