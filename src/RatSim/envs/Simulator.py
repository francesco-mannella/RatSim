from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from . import JsonToPyBox2D as json2d
from .PID import PID
import time, sys, os, glob
from Box2D import b2ContactListener
import os, shutil
import tempfile

class ContactListener(b2ContactListener):
    def __init__(self, bodies):
        b2ContactListener.__init__(self)
        self.contact_db = {}
        self.bodies = bodies

        for h in bodies.keys():
            for k in bodies.keys():
                self.contact_db[(h, k)]= 0

    def BeginContact(self, contact):
        for name, body in self.bodies.items():
            if body == contact.fixtureA.body:
                bodyA = name
            elif body == contact.fixtureB.body:
                bodyB = name
            
        self.contact_db[(bodyA, bodyB)] = len(contact.manifold.points)

    def EndContact(self, contact):
        for name, body in self.bodies.items():
            if body == contact.fixtureA.body:
                bodyA = name
            elif body == contact.fixtureB.body:
                bodyB = name
            
        self.contact_db[(bodyA, bodyB)] = 0

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass

from IPython.display import Image
import matplotlib.image as mpimg

plt.ioff()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class Box2DSim(object):
    """ 2D physics using box2d and a json conf file
    """
    @staticmethod
    def loadWorldJson(world_file):
        jsw = json2d.load_json_data(world_file)
        return jsw

    def __init__(self, world_file=None, world_dict=None, dt=1/80.0, vel_iters=30, pos_iters=2):
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

        self.contact_listener = ContactListener(bodies)
        
        self.dt = dt
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters
        self.world = world
        self.world.contactListener = self.contact_listener
        self.bodies = bodies
        self.joints = joints
        self.joint_pids = { ("%s" % k): PID(dt=self.dt)
                for k in list(self.joints.keys()) }


    def contacts(self, bodyA, bodyB):
        """Read contacts between two parts of the simulation

        Args:

            bodyA (string): the name of the object A
            bodyB (string): the name of the object B

        Returns:

            (int): number of contacts
        """
        c1 = 0
        c2 = 0
        db =  self.contact_listener.contact_db 
        if (bodyA, bodyB) in db.keys(): 
            c1 = self.contact_listener.contact_db[(bodyA, bodyB)]
        if (bodyB, bodyA) in db.keys(): 
            c2 = self.contact_listener.contact_db[(bodyB, bodyA)]

        return c1 + c2

    def move(self, joint_name, angle):
        """change the angle of a joint

        Args:

            joint_name (string): the name of the joint to move
            angle (float): the new angle position

        """
        pid = self.joint_pids[joint_name]
        pid.setpoint = angle

    def step(self):
        """ A simulation step
        """
        for key in list(self.joints.keys()):
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = (self.joint_pids[key].output)
        self.world.Step(self.dt, self.vel_iters, self.pos_iters)


    def move_body(self, angle, speed):
        """Move the robot

        Must have a "body" object

        Args:

            angle (float): moving angle
            speed (float): speed
        """

        body = self.bodies["body"]

        body.angle += angle
        # body.angle = np.maximum(0.3*np.pi, body.angle)
        # body.angle = np.minimum(-0.3*np.pi, body.angle)
        curr_angle = body.angle
        body.linearVelocity = -speed * np.array(
            [np.cos(curr_angle + np.pi / 2), np.sin(curr_angle + np.pi / 2)]
        )
        body.transform = (body.position, curr_angle)

        pid = self.joint_pids["body_to_head"]
        pid.setpoint += 10 * angle
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class TestPlotter:
    """Plotter of simulations
    Builds a simple matplotlib graphic environment
    and render single steps of the simulation within it

    """

    figure = None

    def __init__(
        self, env, xlim=[-5, 5], ylim=[-6, 3], figsize=None, offline=False, figure=None, colors=None):
        """
        Args:
            env (Box2DSim): a emulator object

        """

        self.env = env
        self.offline = offline
        self.xlim = xlim
        self.ylim = ylim

        if TestPlotter.figure is None:
            if figure is None:
                if figsize is None:
                    self.fig = plt.figure()
                else:
                    self.fig = plt.figure(figsize=figsize)
            else:
                self.fig = figure
            TestPlotter.figure = self.fig
        else:
            self.fig = TestPlotter.figure


        self.ax = None

        self.reset(colors)

    def reset(self, colors=None):

        if self.ax is not None:
            plt.delaxes(self.ax)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.polygons = {}
        for key in self.env.sim.bodies.keys():
            if colors is not None and key in colors.keys():
                self.env.sim.bodies[key].color = colors[key]

            self.polygons[key] = Polygon(
                [[0, 0]],
                ec=self.env.sim.bodies[key].color + [1],
                fc=self.env.sim.bodies[key].color + [1],
                closed=True,
            )

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
        """Run a single emulator step"""

        for key in self.polygons:
            body = self.env.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = np.vstack([body.GetWorldPoint(vercs[x]) for x in range(len(vercs))])
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
