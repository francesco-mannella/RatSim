import numpy as np
from scipy import interpolate
import gym
import RatSim

env = gym.make('RatSim-v0')

stime = 1200
for t in range(stime):
    env.render()
    action = np.zeros(6)
    env.step(action)
