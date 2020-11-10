import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import gym
import RatSim

def normal(x, m, s):
    return np.exp(-0.5*(s**-2)*(x - m)**2)


env = gym.make('RatSim-v0')

stime = 500
t = np.linspace(0, 1, 5000)
s = 0.0015
angle, speed = np.array([np.vstack([[p, a]])*normal(t, m, s).reshape(-1, 1)
                         for p, a, m
                         in zip(np.random.uniform(-.04, .04, 100),
                                np.random.uniform(-6, 6, 100),
                                np.random.rand(100))]).sum(0).T

# %%
action = np.zeros(9)
for t in range(stime):
    env.render("offline")


    # whisker joints
    l = np.array([1.7, 1.3, 0.6, 1.7, 1.3, 0.6, 0])
    action[:7] = 0.2*np.pi*np.sin(100*np.pi*t/5000)*l

    # left whiskers have inverted joints
    action[:3] = - action[:3]

    # rotation velocity
    action[-2] = angle[t]

    # linear velocity
    action[-1] = speed[t]

    env.step(action)
