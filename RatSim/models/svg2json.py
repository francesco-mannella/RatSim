#!/usr/bin/python3
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
import json
import numpy as np

fileroot = "rat"
env_scale = 10
object_colors={
    "box": [0.5, 0.5, 0.5],
    "head": [0.5, 0.2, 0],
    "body": [0.5, 0.2, 0],
    "wl1": [0, 0, 0],
    "wl2": [0, 0, 0],
    "wl3": [0, 0, 0],
    "wr1": [0, 0, 0],
    "wr2": [0, 0, 0],
    "wr3": [0, 0, 0]}

# read the SVG file
doc = minidom.parse(fileroot + '.svg')


paths = {}
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

world["body"] = []
allpoints = np.vstack(allpoints)
min = allpoints.min(0)
max = allpoints.max(0)
mean = allpoints.mean(0)
scale = max - min
ratio = float(scale[1]/scale[0])

ratio = np.array([1, ratio]).reshape(1, -1)
# %%
for obj in paths.keys():
    name = obj
    points = (paths[name] - (min + scale/2))/scale
    points = env_scale*ratio*points
    x, y = (points.max(0) - points.min(0))/2
    vx, vy = (points - [[y, x]]).T

    obj_dict = {
        "angle": 0.0,
        "name": obj,
        "color": object_colors[obj],
        "position": {
            "y": x,
            "x": y
        },
        "type": 0,
        "fixture": [{
            "density": 1,
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

for i, obj in enumerate(world["body"]):
    print(world["body"][i]["name"])

jsn = json.dumps(world, indent=4)
with open(fileroot+"_tmp.json", "w") as json_file:
    json_file.write(jsn)
