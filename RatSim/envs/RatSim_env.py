import os
import numpy as np
import gym
from gym import spaces
from .Simulator import Box2DSim as Sim, TestPlotter
import pkg_resources
from scipy import ndimage


def DefaultRewardFun(observation):
    return np.sum(observation['TOUCH_SENSORS'])


class Box2DSimRatEnv(gym.Env):
    """ A single 2D arm Box2DSimwith a box-shaped object
    """
    metadata = {'render.modes': ['human', 'offline']}
    robot_parts_names = ["body", "head", "wl1", "wl2", "wl3", "wr1",
                         "wr2", "wr3"]
    joint_names = ["head_to_wl1", "head_to_wl2", "head_to_wl3", "head_to_wr1",
                   "head_to_wr2", "head_to_wr3", "body_to_head"]
    sensors_names = []

    def __init__(self):

        super(Box2DSimRatEnv, self).__init__()

        self.set_seed()

        self.init_worlds()

        self.init_worlds()
        self.num_joints = 7
        self.num_move_degrees = 2
        self.num_touch_sensors = 6
        self.random_mean = np.array([0, 0])
        self.random_std = 5

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(
            np.hstack([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi,
             -np.pi, -np.pi, -np.pi, 0.0]),
            np.hstack([np.pi,  np.pi, np.pi,  np.pi,  np.pi,
             np.pi,  np.pi, np.pi, 1000000.0]),
            [self.num_joints + self.num_move_degrees], dtype=float)

        self.observation_space = gym.spaces.Dict({
            "JOINT_POSITIONS": gym.spaces.Box(
                -np.inf, np.inf,
                [self.num_joints],
                dtype=float),
            "TOUCH_SENSORS": gym.spaces.Dict(
                {obj_name: gym.spaces.Box(0, np.inf,
                                          [self.num_touch_sensors],
                                          dtype=float)
                 for obj_name in self.object_names}),
            "OBJ_POSITION": gym.spaces.Box(
                -np.inf, np.inf,
                [len(self.object_names), 2],
                dtype=float)})

        self.rendererType = TestPlotter
        self.renderer = None
        self.renderer_figsize = (3, 3)

        self.taskspace_xlim = [-5, 5]
        self.taskspace_ylim = [-6, 3]

        self.set_reward_fun()

        self.set_world(0)
        self.reset()

    def set_renderer_figsize(self, figsize):
        self.renderer_figsize = figsize

    def init_worlds(self):
        self.world_files = [pkg_resources.resource_filename(
            'RatSim', 'models/rat.json')]
        self.worlds = {"rat": 0}
        self.world_object_names = {0: ["box"]}
        self.object_names = self.world_object_names[self.worlds["rat"]]

    def set_world(self, world_id=None):

        self.world_file = self.world_files[0]
        self.object_names = self.world_object_names[self.worlds["rat"]]

        world_dict = Sim.loadWorldJson(self.world_file)
        self.sim = Sim(world_dict=world_dict)

    def set_seed(self, seed=None):
        self.seed = seed
        if self.seed is None:
            self.seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(self.seed)

    def set_reward_fun(self, rew_fun=None):

        self.reward_fun = rew_fun
        if self.reward_fun is None:
            self.reward_fun = DefaultRewardFun

    def set_action(self, action):

        assert(len(action) == self.num_joints + self.num_move_degrees)
        action = np.hstack(action)
        # do action
        for j, joint in enumerate(self.joint_names):
            self.sim.move(joint, action[j])

        direction = action[-2]
        speed = action[-1]
        self.sim.move_body(direction, speed)
        self.sim.step()

    def get_observation(self):

        joints = np.array([self.sim.joints[name].angle
                           for name in self.joint_names])
        sensors = np.array([np.sum([self.sim.contacts(sensor_name, object_name)
                                    for object_name in self.object_names])
                            for sensor_name in self.sensors_names])
        obj_pos = np.array([[self.sim.bodies[object_name].worldCenter]
                            for object_name in self.object_names])
        return joints, sensors, obj_pos

    def sim_step(self, action):

        self.set_action(action)
        joints, sensors, obj_pos = self.get_observation()

        observation = {
            "JOINT_POSITIONS": joints,
            "TOUCH_SENSORS": sensors,
            "OBJ_POSITION": obj_pos}

        return observation

    def step(self, action):

        observation = self.sim_step(action)

        # compute reward
        reward = self.reward_fun(observation)

        # compute end of task
        done = False

        # other info
        info = {}

        return observation, reward, done, info

    def choose_worldfile(self, world_id=None):

        self.world_id = world_id
        if self.world_id is None:
            self.world_id = self.rng.randint(0, len(self.world_files))

        self.world_file = self.world_files[self.world_id]

    def reset(self):

        if self.renderer is not None:
            self.renderer.reset()

        return self.sim_step(np.zeros(self.num_joints + self.num_move_degrees))

    def render(self, mode='human'):

        if mode == 'human':
            if self.renderer is None:
                self.renderer = self.rendererType(
                    self,
                    xlim=self.taskspace_xlim,
                    ylim=self.taskspace_ylim,
                    figsize=self.renderer_figsize)
        elif mode == 'offline':
            if self.renderer is None:
                self.renderer = self.rendererType(
                    self,
                    xlim=self.taskspace_xlim,
                    ylim=self.taskspace_ylim,
                    offline=True,
                    figsize=self.renderer_figsize)
        self.renderer.step()
