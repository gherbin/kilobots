import numpy as np

class Shape:

    def __init__(self, scale, map):
        self.scale = scale
        self.map = map

    def __str__(self):
        str_map = "\n".join([str(row) for row in np.rot90(self.map)])
        return "scale: " + str(self.scale) + "\n" + str_map