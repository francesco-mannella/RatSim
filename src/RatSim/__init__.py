from gym.envs.registration import register

register(id='RatSim-v0',
    entry_point='RatSim.envs:Box2DSimRatEnv',
)


from RatSim.envs import Box2DSimRatEnv
