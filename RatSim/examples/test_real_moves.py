import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import gym
import RatSim

rng = np.random.RandomState(4)

def normal(x, m, s):
    return np.exp(-0.5*(s**-2)*(x - m)**2)


env = gym.make('RatSim-v0')

stime = 500

t = np.linspace(0, 1, stime)
s = 0.005
peaks_num = 20
peaks = rng.uniform(0, 1, peaks_num)
amplitudes = rng.uniform(-13, 15, peaks_num)
angles = rng.uniform(-.04, .04, peaks_num)

angles, speeds = np.array([[[angle, ampl]]*normal(t, mean, s).reshape(-1, 1)
                         for angle, ampl, mean
                         in zip(angles, amplitudes, peaks)]).sum(0).T

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(211)
ax1.set_title("Linear velocity in time")
ax1.set_xticks([])
splot, = ax1.plot(t, speeds, alpha=0.5, lw=4)
ax2 = fig.add_subplot(212)
ax2.set_title("Direction in time")
aplot, = ax2.plot(t, angles, alpha=0.5,lw=4)

plt.show()
# %%
for t in range(stime):
    
    #env.render("offline")
    env.render()

    action = np.zeros(15)
    # whisker amplitudes
    amplitudes = 0.2*np.pi*np.array([1.7, 1.3, 0.6, 1.7, 1.3, 0.6])
    action[:6] = amplitudes

    # rotation velocity
    action[-2] = angles[t]

    # linear velocity
    action[-1] = speeds[t]

    state, *_ = env.step(action)
    if np.sum(state["TOUCH_SENSORS"]):
        print(state["TOUCH_SENSORS"])

