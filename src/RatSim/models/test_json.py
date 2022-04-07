import json
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
filePathName = "rat_tmp.json"

with open(filePathName, "r") as json_file:
    jsw = json.load(json_file)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, aspect="equal")
ax.set_xlim([0, 10])
ax.set_ylim([0, 6])
for obj in jsw["body"]:
    ax.set_title(obj["name"])
    x = np.array(obj["fixture"][0]["polygon"]["vertices"]["x"])
    x += obj["position"]["x"]
    y = np.array(obj["fixture"][0]["polygon"]["vertices"]["y"])
    y += obj["position"]["y"]

    ax.plot(x, y)
    input()
