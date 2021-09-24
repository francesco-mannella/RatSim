from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from . import JsonToPyBox2D as json2d
from .PID import PID
import time
import sys
import os
import glob

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Box2DSim(object):
    """ 2D physics using box2d and a json conf file
    """
    @staticmethod
    def loadWorldJson(world_file):
        jsw = json2d.load_json_data(world_file)
        return jsw

    def __init__(self, world_file=None, world_dict=None, dt=1/80.0,
                 vel_iters=30, pos_iters=2):
        """
        Args:

            world_file (string): the json file from which all objects are created
            world_dict (dict): the json object from which all objects are created
            dt (float): the amount of time to simulate, this should not vary.
            pos_iters (int): for the velocity constraint solver.
            vel_iters (int): for the position constraint solver.

        """
        if world_file is not None:
            world, bodies, joints = json2d.createWorldFromJson(world_file)
        else:
            world, bodies, joints = json2d.createWorldFromJsonObj(world_dict)
        self.dt = dt
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters
        self.world = world
        self.bodies = bodies
        self.joints = joints
        self.joint_pids = {("%s" % k): PID(dt=self.dt)
                           for k in list(self.joints.keys())}

    def contacts(self, bodyA, bodyB):
        """ Read contacts between two parts of the simulation

        Args:

            bodyA (string): the name of the object A
            bodyB (string): the name of the object B

        Returns:

            (int): number of contacts
        """


        contacts = 0
        for ce in self.bodies[bodyA].contacts:
            if ce.contact.touching is True:
                if ce.contact.fixtureA.body == self.bodies[bodyB]:
                    contacts += 1
        return contacts

    def move(self, joint_name, angle):
        """ change the angle of a joint

        Args:

            joint_name (string): the name of the joint to move
            angle (float): the new angle position

        """
        pid = self.joint_pids[joint_name]
        pid.setpoint = angle

    def move_body(self, angle, speed):
        """ Move the robot

        Must have a "body" object

        Args:

            angle (float): moving angle
            speed (float): speed
        """

        body = self.bodies["body"]

        body.angle += angle
        #body.angle = np.maximum(0.3*np.pi, body.angle)
        #body.angle = np.minimum(-0.3*np.pi, body.angle)
        curr_angle = body.angle
        body.linearVelocity = -speed*np.array(
            [np.cos(curr_angle+np.pi/2), np.sin(curr_angle+np.pi/2)])
        body.transform = (body.position, curr_angle)

        pid = self.joint_pids["body_to_head"]
        pid.setpoint += 10*angle

    def step(self):
        """ A simulation step
        """
        for key in list(self.joints.keys()):
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = (self.joint_pids[key].output)
        self.world.Step(self.dt, self.vel_iters, self.pos_iters)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class TestPlotter:
    """ Plotter of simulations
    Builds a simple matplotlib graphic environment
    and render single steps of the simulation within it

    """

    def __init__(self, env, xlim=[-5, 5], ylim=[-6, 3],
                 figsize=None, offline=False):
        """
        Args:
            env (Box2DSim): a emulator object

        """

        self.env = env
        self.offline = offline
        self.xlim = xlim
        self.ylim = ylim

        if figsize is None:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure(figsize=figsize)

        self.ax = None

        self.reset()

    def reset(self):

        if self.ax is not None:
            plt.delaxes(self.ax)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.polygons = {}
        for key in self.env.sim.bodies.keys():
            self.polygons[key] = Polygon(
                [[0, 0]],
                ec=self.env.sim.bodies[key].color + [1],
                fc=self.env.sim.bodies[key].color + [1],
                closed=True)

            self.ax.add_artist(self.polygons[key])

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        if not self.offline:
            self.fig.show()
        else:
            self.ts = 0

    def onStep(self):
        pass

    def step(self):
        """ Run a single emulator step
        """

        for key in self.polygons:
            body = self.env.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = np.vstack([body.GetWorldPoint(vercs[x])
                              for x in range(len(vercs))])
            self.polygons[key].set_xy(data)

        self.onStep()

        if not self.offline:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        else:
            if not os.path.exists("frames"):
                os.makedirs("frames")

            self.fig.savefig("frames/frame%06d.png" % self.ts, dpi=200)
            self.fig.canvas.draw()
            self.ts += 1
