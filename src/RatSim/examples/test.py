import gym
import RatSim

import numpy as np
rng = np.random.RandomState(2)

env = gym.make('RatSim-v0')
action = np.array(
    [0.5*0.45*np.pi,   # head_to_wl1 amp
     1.5*0.45*np.pi,   # head_to_wl2 amp
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

observation = env.reset(0)
for t in range(30000):


    #env.render("offline")
    if t%50 == 0:
        env.render("human")
    action[-2] += 0.0002*(0.04*rng.randn() - action[-2])   # random angular velocity
    action[-1] += 0.0002*(0.1*rng.randn() - action[-1])  # randomlinear velocity

    observation,*_ = env.step(action)
    print(observation["TOUCH_SENSORS"])
    if t < 1300:
        env.move_object("box", [0.0, 0.001])
