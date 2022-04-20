#!/usr/bin/python3
import matplotlib.pyplot as plt
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
import json
import numpy as np

fileroot = "rat"
env_scale = 15
object_colors = {
    "box": [0.5, 0.5, 0.5],
    "head": [0.5, 0.2, 0],
    "body": [0.5, 0.2, 0],
    "wl1": [0, 0, 0],
    "wl2": [0, 0, 0],
    "wl3": [0, 0, 0],
    "wr1": [0, 0, 0],
    "wr2": [0, 0, 0],
    "wr3": [0, 0, 0]}
object_masses = {
    "box": 40,
    "head": 1,
    "body": 40,
    "wl1": 0.01,
    "wl2": 0.01,
    "wl3": 0.01,
    "wr1": 0.01,
    "wr2": 0.01,
    "wr3": 0.01}
joint_torques = {
    "head_to_wl1": 10,
    "head_to_wl2": 10,
    "head_to_wl3": 10,
    "head_to_wr1": 10,
    "head_to_wr2": 10,
    "head_to_wr3": 10,
    "body_to_head": 10000}

# read the SVG file
doc = minidom.parse(fileroot + '.svg')


paths = {}
joints = {}
allpoints = []
for path in doc.getElementsByTagName('path'):
    id = path.getAttribute('id')
    d = path.getAttribute('d')
    points = []
    for e in parse_path(d):
        if isinstance(e, Line):
            x0 = e.start.real
            y0 = e.start.imag
            x1 = e.end.real
            y1 = e.end.imag
            points.append([x0, y0])
    #
    points.append([x1, y1])
    points.append(points[0])
    paths[id] = np.array(points)
    allpoints.append(paths[id])

for joint in doc.getElementsByTagName('circle'):
    id = joint.getAttribute("id")
    objA, objB = id.split("_to_")
    cx = abs(float(joint.getAttribute("cx")))
    cy = abs(float(joint.getAttribute("cy")))

    jInfo = (objA, objB, cx, cy)

    if objA in joints.keys():
        joints[objA].append(jInfo)
    else:
        joints[objA] = [jInfo]


doc.unlink()

world = {
    "gravity": {
        "y": 0,
        "x": 0
    },
    "autoClearForces": True,
    "continuousPhysics": True,
    "subStepping": False,
    "warmStarting": True
}

allpoints = np.vstack(allpoints)
min = allpoints.min(0)
max = allpoints.max(0)
mean = allpoints.mean(0)
scale = max - min
ratio = float(scale[1]/scale[0])
ratio = np.array([1, ratio]).reshape(1, -1)

# %%
plt.ion()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect="equal")

sc1 = ax.scatter(0, 0, s=100, c="r")
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)


def rescale(name, other_points=None):

    points = (paths[name] - (min + scale/2))/scale
    points = env_scale*ratio*points
    x, y = points.min(0) + (points.max(0) - points.min(0))/2
    vx, vy = (points - [[x, y]]).T
    if other_points is None:
        ax.scatter(x, y, s=200, c="k")
        ax.plot(x+vx, y+vy, c="k")
    if other_points is not None:
        op = (other_points - (min + scale/2))/scale
        op = env_scale*ratio*op
        vox, voy = (op - [[x, y]]).T
        sc1.set_offsets(np.vstack([x+vox, y+voy]).T)
        print(name, x, y, vox, voy)
        plt.pause(0.1)
        return x, y, vx, vy, vox[0], voy[0]
    else:
        return x, y, vx, vy


world["body"] = []
for obj in paths.keys():
    name = obj
    x, y, vx, vy = rescale(name)

    obj_dict = {
        "angle": 0.0,
        "name": obj,
        "color": object_colors[obj],
        "position": {
            "x": x,
            "y": y
        },
        "type": 2,
        "fixture": [{
            "density": object_masses[obj],
            "group_index": 0,
            "polygon": {
                "vertices": {
                    "x": vx.tolist(),
                    "y": vy.tolist()
                }
            }
        }]
    }
    world["body"].append(obj_dict)


body_idcs = {}
for i, obj in enumerate(world["body"]):
    print(world["body"][i]["name"])
    body_idcs[world["body"][i]["name"]] = i

world["joint"] = []
for joint_set in joints.values():
    for joint in joint_set:
        nameA, nameB, cx, cy = joint

        # point c wrt A
        *_, cAx, cAy = rescale(nameA, np.array([[cx, cy]]))
        # point c wrt B
        *_, cBx, cBy = rescale(nameB, np.array([[cx, cy]]))

        jont_dict = {
            "name": nameA+"_to_"+nameB,
            "type": "revolute",
            "bodyA": body_idcs[nameA],
            "bodyB": body_idcs[nameB],
            "jointSpeed": 0,
            "refAngle": 0,
            "collideConnected": False,
            "maxMotorTorque": joint_torques[nameA+"_to_"+nameB],
            "enableLimit": True,
            "motorSpeed": 0,
            "anchorA": {
                    "x": cAx,
                    "y": cAy
            },
            "anchorB": {
                "x": cBx,
                "y": cBy
            },
            "upperLimit": 3.14,
            "lowerLimit": -3.14,
            "enableMotor": True
        }
        world["joint"].append(jont_dict)


for i, joint in enumerate(world["joint"]):
    print(world["joint"][i]["name"])

jsn = json.dumps(world, indent=4)
with open(fileroot+".json", "w") as json_file:
    json_file.write(jsn)
