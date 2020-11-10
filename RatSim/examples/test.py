import numpy as np
from scipy import interpolate
import gym
import RatSim

env = gym.make('RatSim-v0')

stime = 1200
action = np.zeros(9)
for t in range(stime):
    env.render()
    action[:7] = np.sin(15*np.pi*t/stime)*np.ones(7)
    action[6] = 0
    action[:3] = - action[:3]
    action[-2] = 5000
    action[-1] = 1
    env.step(action)
