import gym
import RatSim

import numpy as np
rng = np.random.RandomState(2)

env = gym.make('RatSim-v0')
action = np.array(
    [1.0*0.45*np.pi,   # head_to_wl1 amp
     0.5*0.45*np.pi,   # head_to_wl2 amp
     0.5*0.45*np.pi,   # head_to_wl3 amp
     1.0*0.45*np.pi,   # head_to_wr1 amp
     0.5*0.45*np.pi,   # head_to_wr2 amp
     0.5*0.45*np.pi,   # head_to_wr3 amp
     0,
     0,
     0,
     0,
     0,
     0,
     0.0,   # body_to_head angle
     0,     # angular velocity
     0])     # linear velocity

for t in range(300):

    #env.render("offline")
    env.render()
    action[-2] += 0.02*(0.04*rng.randn() - action[-2])   # random angular velocity
    action[-1] += 0.02*(10*rng.randn() - action[-1])  # randomlinear velocity

    observation = env.step(action)
    env.moveObjext("box", [0.0, 0.001])
